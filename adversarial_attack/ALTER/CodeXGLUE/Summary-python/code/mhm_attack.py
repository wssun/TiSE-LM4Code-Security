import sys
import os

from tqdm import tqdm

sys.path.append('../../../')
sys.path.append('../../../python_parser')

import csv
import json
import argparse
import warnings
import torch
import numpy as np
from model import Model, CodeLlama
from utils import set_seed
from utils import Recorder
from run import TextDataset
from utils import CodeDataset
from python_parser.parser_folder import remove_comments_and_docstrings
from run_parser import get_identifiers
from transformers import RobertaForMaskedLM
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from attacker import MHM_Attacker, convert_code_to_features_codellama
from attacker import convert_code_to_features

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning\

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'codellama': (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    # 'codellama': (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer)
}

from utils import build_vocab

def get_device(args):
    device = f'cuda:{args.gpu_input}' if torch.cuda.is_available() and int(
        args.gpu_input) < torch.cuda.device_count() else 'cpu'
    return device


def main():
    import json
    import pickle
    import time
    import os

    # import tree as Tree
    # from dataset import Dataset, POJ104_SEQ
    # from lstm_classifier import LSTMEncoder, LSTMClassifier

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default='train_subs.jsonl', type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default='./saved_models', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default='../preprocess/dataset/python_human.jsonl', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="codellama", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default='../../../../CodeLlama-7b-Instruct-hf', type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--base_model", default='../../../../CodeLlama-7b-Instruct-hf', type=str,
                        help="Base Model")
    parser.add_argument("--store_path", default='../preprocess/dataset/java_human_result.jsonl', type=str,
                        help="results")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default='../../../../CodeLlama-7b-Instruct-hf', type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--original", action='store_true',
                        help="Whether to MHM original.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--gpu_input", default="1", type=str)

    args = parser.parse_args()

    args.device = get_device(args)
    # 检查输出文件是否存在，若存在则删除
    output = args.store_path
    if os.path.exists(output):
        os.remove(output)

    args.start_epoch = 0
    args.start_step = 0

    ## Load Target Model
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')  # 读取model的路径
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        # 如果路径存在且有内容，则从checkpoint load模型
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())
        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    # config.num_labels = 2 # 只有一个label?
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=False,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    vocab = tokenizer.get_vocab()
    args.yes_token_id = vocab['▁Yes']
    args.no_token_id = vocab['▁No']

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.base_model,
            from_tf=bool('.ckpt' in args.model_name_or_path),
            config=config,
            torch_dtype=torch.float16,
            # load_in_4bit=True,
            # device_map='auto',
            use_safetensors=False,
            cache_dir=args.cache_dir if args.cache_dir else None).to(args.device)
    else:
        model = model_class(config)

    # Load Dataset
    ## Load Dataset
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    # Load original source codes
    source_codes = []
    generated_substitutions = []
    with open(args.eval_data_file) as f:
        for line in f:
            js = json.loads(line.strip())
            # code = ' '.join(js['func'].split())
            code = js['func']
            source_codes.append(code)
            generated_substitutions.append(js['substitutes'])
    assert (len(source_codes) == len(eval_dataset) == len(generated_substitutions))

    code_tokens = []
    for index, code in enumerate(source_codes):
        code_tokens.append(get_identifiers(code, "c")[1])

    id2token, token2id = build_vocab(code_tokens, 5000)

    model = CodeLlama(model, config, tokenizer, args)

    attacker = MHM_Attacker(args, model, model, tokenizer, token2id, id2token)

    # token2id: dict,key是变量名, value是id
    # id2token: list,每个元素是变量名

    print("ATTACKER BUILT!")

    adv = {"tokens": [], "raw_tokens": [], "ori_raw": [],
           'ori_tokens': [], "label": [], }
    n_succ = 0.0
    total_cnt = 0
    query_times = 0
    all_start_time = time.time()
    for index, example in tqdm(enumerate(eval_dataset)):
        code = source_codes[index]
        substituions = generated_substitutions[index]
        identifiers, code_tokens = get_identifiers(code, lang='python')
        code_tokens = [i for i in code_tokens]
        processed_code = " ".join(code_tokens)

        # new_feature = convert_code_to_features_codellama(processed_code, tokenizer, example[1].item(), args)
        # new_dataset = CodeDataset([new_feature])

        orig_prob = model.get_results([example], args.eval_batch_size)
        # orig_label = orig_label[0]
        ground_truth = example[1]
        label = example[1]
        # if orig_label != ground_truth:
        #     label = orig_label
        # else:
        #     label = ground_truth

        start_time = time.time()

        # 这里需要进行修改.
        if args.original:
            _res = attacker.mcmc_random(tokenizer, substituions, code,
                                        _label=label, _n_candi=30,
                                        _max_iter=100, _prob_threshold=1)
        else:
            _res, code_candidate = attacker.mcmc(tokenizer, substituions, code,
                                 _label=label, _n_candi=30,
                                 _max_iter=20, _prob_threshold=1, _orig_prob = orig_prob)
        code_candidate = sorted(code_candidate, key=lambda x: x[1])
        adv_list = {}
        adv_list['code_list'] = code_candidate
        adv_list['label'] = ground_truth
        with open(args.store_path, "a") as wf:
            wf.write(json.dumps(adv_list) + '\n')
        if _res['succ'] is None:
            continue
        if _res['succ'] == True:
            print("EXAMPLE " + str(index) + " SUCCEEDED!")
            n_succ += 1
            adv['tokens'].append(_res['tokens'])
            adv['raw_tokens'].append(_res['raw_tokens'])
        else:
            print("EXAMPLE " + str(index) + " FAILED.")
        total_cnt += 1
        print("  time cost = %.2f min" % ((time.time() - start_time) / 60))
        time_cost = (time.time() - start_time) / 60
        print("  ALL EXAMPLE time cost = %.2f min" % ((time.time() - all_start_time) / 60))
        print("  curr succ rate = " + str(n_succ / total_cnt))
        print("Query times in this attack: ", model.query - query_times)
        print("All Query times: ", model.query)

        query_times = model.query


if __name__ == "__main__":
    main()