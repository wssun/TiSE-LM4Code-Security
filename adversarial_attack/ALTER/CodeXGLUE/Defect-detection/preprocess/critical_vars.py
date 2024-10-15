import json
import sys
import torch
import argparse
from tqdm import tqdm
import re
from sklearn.cluster import KMeans
from collections import Counter
import jsonlines
sys.path.append('../../../')
sys.path.append('../../../python_parser')
import os
# from attacker import
from python_parser.run_parser import get_identifiers, remove_comments_and_docstrings
from utils import is_valid_variable_name, _tokenize, get_identifier_posistions_from_code, get_masked_code_by_position, \
    get_substitues, is_valid_substitue
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from utils import CodeDataset

sys.path.append('../code')
from model import CodeLlama

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx=str(idx)
        self.label=label
    def get_input_tokens(self):
        """获取 input_tokens"""
        return self.input_tokens
    def get_input_ids(self):
        """获取 input_ids"""
        return self.input_ids

    def set_input_tokens(self, new_tokens):
        """设置新的 input_tokens"""
        self.input_tokens = new_tokens

    def set_input_ids(self, new_ids):
        """设置新的 input_ids"""
        self.input_ids = new_ids


def get_device(args):
    device = f'cuda:{args.gpu_input}' if torch.cuda.is_available() and int(
        args.gpu_input) < torch.cuda.device_count() else 'cpu'
    return device

def split_code(code):
    # 使用正则表达式匹配单词和符号
    tokens = re.findall(r"[\w']+|[.,!?;()\"']", code)
    return tokens
def convert_examples_to_features_codellama(code, label, tokenizer):
    # 将源代码紧凑化'join'和'split'
    code=' '.join(code.split())
    # pro_code = get_prompt(code,system_prompt, prompt)
    code_tokens=tokenizer.tokenize(code)
    source_tokens = code_tokens
    # code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    # source_tokens = [tokenizer.bos_token] + code_tokens + [tokenizer.eos_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    # padding_length = args.block_size - len(source_ids)
    # source_ids+=[tokenizer.pad_token_id]*padding_length
    return torch.tensor(source_ids), torch.tensor(label)

def convert_code_to_dataset(code, tokenizer, label):
    code=' '.join(code.split())
    code_tokens=tokenizer.tokenize(code)
    source_tokens = code_tokens
    # code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    # source_tokens = [tokenizer.bos_token] + code_tokens + [tokenizer.eos_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    # padding_length = args.block_size - len(source_ids)
    # source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids, 0, label)
def get_importance_score(example, words_list: list, variable_names: list, tgt_model, tokenizer):
    '''Compute the importance score of each variable'''
    # label: example[1] tensor(1)
    # 1. 过滤掉所有的keywords.
    positions = get_identifier_posistions_from_code(words_list, variable_names)
    # 需要注意大小写.
    if len(positions) == 0:
        ## 没有提取出可以mutate的position
        return None, None, None

    new_example = []

    # 2. 得到Masked_tokens
    masked_token_list, replace_token_positions = get_masked_code_by_position(words_list, positions)
    # replace_token_positions 表示着，哪一个位置的token被替换了.

    # 需要考虑原本替换的符号占几位
    key_num = {}
    weighted_list = [0]
    for v in variable_names:
        code_tokens = tokenizer.tokenize(v)
        key_num[v] = len(code_tokens)
    for key, values in positions.items():
        weighted_list.extend([key_num[key]] * len(values))

    for index, tokens in enumerate([words_list] + masked_token_list):
        new_code = ' '.join(tokens)
        new_feature = convert_code_to_dataset(new_code, tokenizer, example[1].item())
        # 对于CodeLlamaTokenizerFast，<unk>前后会识别为“_”,所以删掉
        # remove_around_symbol(new_feature, '<unk>', weighted_list[index])
        new_example.append(new_feature)
    # 断言检查是否都同样长度
    # assert_equal_length(new_example)
    new_dataset = CodeDataset(new_example)
    # 3. 将他们转化成features
    logits, preds = tgt_model.get_results(new_dataset, 1)
    # logits, preds = tgt_model.get_results(new_dataset, 1)
    orig_probs = logits[0]
    orig_label = preds[0]
    # 第一个是original code的数据.

    orig_prob = max(orig_probs)
    # predicted label对应的probability

    importance_score = []
    for prob in logits[1:]:
        importance_score.append(orig_prob - prob[orig_label])

    return importance_score, replace_token_positions, positions



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_data_file", default='./dataset/test_codellama_unixcoder_5.jsonl', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--base_model",
                        # default='../../../codebert-base-mlm',
                        default='../../../../CodeLlama-7b-Instruct-hf',
                        type=str,
                        help="Base Model")
    parser.add_argument("--store_path", default='./dataset/adv_defect_3/0_25_icl_subs.jsonl', type=str,
                        help="results")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--index", default=[1, 4], nargs='+',
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--mutation", default=False, type=bool,
                        help="use mutation method or not")
    parser.add_argument("--candidate_num", default=40, type=int,
                        help="need be odd")
    parser.add_argument("--gpu_input", default="0", type=str)
    parser.add_argument("--icl_number", default=3, type=int)
    args = parser.parse_args()

    args.device = get_device(args)



    eval_data = []

    # codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    config = AutoConfig.from_pretrained(args.base_model)
    codebert_mlm = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            config=config,
            torch_dtype=torch.float16,
            # load_in_4bit=True,
            # device_map='auto',
            use_safetensors=False,
        ).to(args.device)
    # tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)
    tokenizer_mlm = AutoTokenizer.from_pretrained(args.base_model)
    # codebert_mlm.to('cuda')

    vocab = tokenizer_mlm.get_vocab()
    args.yes_token_id = vocab['▁Yes']
    args.no_token_id = vocab['▁No']

    model = CodeLlama(codebert_mlm, config, tokenizer_mlm, args)

    query_list = []
    with open(args.eval_data_file) as rf:
        for i, line in enumerate(rf):
            if i < int(args.index[0]) or i >= int(args.index[1]):
                continue
            item = json.loads(line.strip())
            query_list.append((item['code_tokens'], item['docstring_tokens']))
            for j in item['code_candidates_tokens']:
                candidate_data = {}
                candidate_data['func'] = j['ori_code']
                candidate_data['target'] = j['ori_docstring']
                eval_data.append(candidate_data)
                if j['idx'] == args.icl_number: break
    print(len(eval_data))

    # 从store_path中提取数字信息
    numbers_in_path = re.search(r'\d+_\d+', args.store_path)
    if numbers_in_path:
        summary_filename = f'{numbers_in_path.group()}'
    else:
        summary_filename = 'test'  # 如果没有匹配到数字，使用默认文件名

    # 获取jsonl文件的目录路径
    directory_path = os.path.dirname(args.store_path)

    # 构建完整的文件路径
    results_filepath = os.path.join(directory_path, summary_filename)

    # 确保文件夹存在
    os.makedirs(directory_path, exist_ok=True)

    label_0_vars = []
    label_1_vars = []

    for item in tqdm(eval_data):
        identifiers, code_tokens = get_identifiers(remove_comments_and_docstrings(item["func"], "c"), "c")
        processed_code = " ".join(code_tokens)

        words, sub_words, keys = _tokenize(processed_code, tokenizer_mlm)

        variable_names = []
        for name in identifiers:
            if ' ' in name[0].strip():
                continue
            variable_names.append(name[0])

        # 似乎codellama生成embedding不可以加特殊token（bos，eos等）？？
        # sub_words = [tokenizer_mlm.bos_token] + sub_words[:args.block_size - 2] + [tokenizer_mlm.eos_token]
        sub_words = sub_words[:args.block_size]

        input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])
        # input_ids_ = tokenizer_mlm(sub_words, return_tensors='pt', add_special_tokens=False)

        word_predictions = codebert_mlm(input_ids_.to(args.device))[0].squeeze()  # seq-len(sub) vocab
        word_pred_scores_all, word_predictions = torch.topk(word_predictions, args.candidate_num, -1)  # seq-len k
        # 得到前k个结果.

        word_predictions = word_predictions[1:len(sub_words) + 1, :]
        word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]

        names_positions_dict = get_identifier_posistions_from_code(words, variable_names)

        variable_substitue_dict = {}
        with torch.no_grad():
            orig_embeddings = codebert_mlm.model(input_ids_.to(args.device))[0]

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        example = convert_examples_to_features_codellama(item["func"], item["target"], tokenizer_mlm)

        importance_score, replace_token_positions, names_positions_dict = get_importance_score(example,
                                                                                               words,
                                                                                               variable_names,
                                                                                               model,
                                                                                               tokenizer_mlm,
                                                                                               )
        token_pos_to_score_pos = {}

        if importance_score is None: continue

        for i, token_pos in enumerate(replace_token_positions):
            token_pos_to_score_pos[token_pos] = i
        # 重新计算Importance score，将所有出现的位置加起来（而不是取平均）.
        names_to_importance_score = {}

        for name in names_positions_dict.keys():
            total_score = 0.0
            positions = names_positions_dict[name]
            for token_pos in positions:
                # 这个token在code中对应的位置
                # importance_score中的位置：token_pos_to_score_pos[token_pos]
                total_score += importance_score[token_pos_to_score_pos[token_pos]]

            names_to_importance_score[name] = total_score
        # 根据importance_score进行排序
        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=True)
        saved_vars = [item[0] for item in sorted_list_of_names[:5] if isinstance(item[0], str)]
        if item['target'] == 0:
            label_0_vars = saved_vars + label_0_vars
        elif item['target'] == 1:
            label_1_vars = saved_vars + label_1_vars


    label_1_counts = Counter(label_1_vars)
    label_0_counts = Counter(label_0_vars)


    with open(results_filepath + '_label_0.jsonl', "w") as wf:
            wf.write(json.dumps(label_0_counts) + '\n')
    with open(results_filepath + '_label_1.jsonl', "w") as wf:
            wf.write(json.dumps(label_1_counts) + '\n')

if __name__ == "__main__":
    main()
