# coding=utf-8
# @Time    : 2020/8/13
# @Author  : Zhou Yang
# @Email   : zyang@smu.edu.sg
# @File    : gi_attack.py
'''For attacking CodeBERT models'''
import sys
import os

sys.path.append('../../../')
sys.path.append('../../../python_parser')
import re
import json
import logging
import argparse
import warnings
import torch
import time
from model import Model
from run import TextDataset
from utils import set_seed
from python_parser.parser_folder import remove_comments_and_docstrings
from utils import Recorder
from attacker import Attacker
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from model import CodeLlama
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'codellama': (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    # 'codellama': (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer)
}

logger = logging.getLogger(__name__)



def get_device(args):
    device = f'cuda:{args.gpu_input}' if torch.cuda.is_available() and int(
        args.gpu_input) < torch.cuda.device_count() else 'cpu'
    return device

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default='train_subs.jsonl', type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default='./saved_models', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default='../preprocess/dataset/test_icl_subs.jsonl', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="codellama", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default='../../../../CodeLlama-7b-Instruct-hf', type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--base_model", default='../../../../CodeLlama-7b-Instruct-hf', type=str,
                        help="Base Model")
    parser.add_argument("--store_path", default='../preprocess/dataset/test_attack_genetic.jsonl', type=str,
                        help="Path to store the CSV file")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="../../../../CodeLlama-7b-Instruct-hf", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--block_size", default=4096, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--use_ga", default=False, action='store_true',
                        help="Whether to GA-Attack.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Batch size p er GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--gpu_input", default="0", type=str)
    parser.add_argument("--query_filename", default='../preprocess/dataset/adv_defect_3/0_50_icl_query.jsonl', type=str)

    args = parser.parse_args()

    args.device = get_device(args)
    # Set seed
    set_seed(args.seed)

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

    ## Load CodeBERT (MLM) model
    # codebert_mlm = model
    # tokenizer_mlm = tokenizer
    # codebert_mlm.to('cuda')

    model = CodeLlama(model, config, tokenizer, args)

    # checkpoint_prefix = 'checkpoint-best-acc/model.bin'
    # output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    # model.load_state_dict(torch.load(output_dir))
    # model.to(args.device)



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


    # 从store_path中提取数字信息
    numbers_in_path = re.search(r'\d+_\d+', args.store_path)
    if numbers_in_path:
        summary_filename = f'{numbers_in_path.group()}_step2_results.txt'
    else:
        summary_filename = 'results_text.txt'  # 如果没有匹配到数字，使用默认文件名

    # 获取jsonl文件的目录路径
    directory_path = os.path.dirname(args.store_path)

    # 构建完整的文件路径
    results_filepath = os.path.join(directory_path, summary_filename)

    # 确保文件夹存在
    os.makedirs(directory_path, exist_ok=True)


    success_attack = 0
    total_cnt = 0

    # recoder = Recorder(args.csv_store_path)

    attacker = Attacker(args, model, tokenizer, model, tokenizer, use_bpe=1, threshold_pred_score=0)
    start_time = time.time()
    query_times = 0
    for index, example in enumerate(eval_dataset):
        example_start_time = time.time()
        code = source_codes[index]
        substituions = generated_substitutions[index]
        code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words = attacker.greedy_attack(example, code, substituions)
        # 保留原始code作为icl结果的ori_label
        adv_code.insert(0, (code,np.float64(0)))
        # attack_type = "Greedy"
        # if is_success == -1 and args.use_ga:
        #     # 如果不成功，则使用gi_attack
        #     code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words = attacker.ga_attack(
        #         example, code, substituions, initial_replace=replaced_words)
        #     attack_type = "GA"

        example_end_time = (time.time() - example_start_time) / 60

        print("Example time cost: ", round(example_end_time, 2), "min")
        print("ALL examples time cost: ", round((time.time() - start_time) / 60, 2), "min")
        # score_info = ''
        # if names_to_importance_score is not None:
        #     for key in names_to_importance_score.keys():
        #         score_info += key + ':' + str(names_to_importance_score[key]) + ','
        #
        # replace_info = ''
        # if replaced_words is not None:
        #     for key in replaced_words.keys():
        #         replace_info += key + ':' + replaced_words[key] + ','
        print("Query times in this attack: ", model.query - query_times)
        print("All Query times: ", model.query)
        adv_list = {}
        adv_list['code_list'] = adv_code
        adv_list['label'] = true_label
        with open(args.store_path, "a") as wf:
            wf.write(json.dumps(adv_list) + '\n')

        query_times = model.query

        if is_success >= -1:
            # 如果原来正确
            total_cnt += 1
        if is_success == 1:
            success_attack += 1

        if total_cnt == 0:
            continue
        print("Success rate: ", 1.0 * success_attack / total_cnt)
        print("Successful items count: ", success_attack)
        print("Total count: ", total_cnt)
        print("Index: ", index)
        print()
    # 循环结束后，将汇总结果写入文件
    with open(results_filepath, 'w') as results_file:
        results_file.write(f"Success rate: {1.0 * success_attack / total_cnt if total_cnt else 0:.2f}\n")
        results_file.write(f"Successful items count: {success_attack}\n")
        results_file.write(f"Total count: {total_cnt}\n")

    print("All processing complete. Results saved to:", results_filepath)

if __name__ == '__main__':
    main()
