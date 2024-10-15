import json
import sys
import copy
import torch
import argparse
from tqdm import tqdm
import os
from sklearn.cluster import KMeans
import Levenshtein
from itertools import combinations
sys.path.append('../../../')
sys.path.append('../../../python_parser')

# from attacker import
from python_parser.run_parser import get_identifiers, remove_comments_and_docstrings
from utils import is_valid_variable_name, _tokenize, get_identifier_posistions_from_code, get_masked_code_by_position, \
     get_substitues, is_valid_substitue
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def find_closest_words(target, words_list, top_n=5):
    # 对所有单词按与目标单词的距离进行排序
    sorted_words = sorted(words_list, key=lambda word: Levenshtein.distance(target, word))
    # 返回距离最近的top_n个单词
    return sorted_words[:top_n]

def generate_combinations(words, max_group_size=3):
    results = []
    # 生成1到max_group_size个单词的组合
    for r in range(1, max_group_size+1):
        # 生成所有可能的组合
        all_combinations = list(combinations(words, r))
        # 将每个组合的单词用下划线连接成字符串
        for comb in all_combinations:
            results.append('_'.join(comb))
    return results



def get_device(args):
    device = f'cuda:{args.gpu_input}' if torch.cuda.is_available() and int(
        args.gpu_input) < torch.cuda.device_count() else 'cpu'
    return device

def read_jsonl_to_dict(file_path):
    """读取JSONL文件到字典。"""
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            json_data = json.loads(line.strip())
            data.update(json_data)
    return data



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_data_file", default='./dataset/test_codellama_unixcoder_5.jsonl', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--base_model",
                        # default='../../../codebert-base-mlm',
                        default='../../../../CodeLlama-7b-Instruct-hf',
                        type=str,
                        help="Base Model")
    parser.add_argument("--store_path", default='./dataset/test_icl_subs.jsonl', type=str,
                        help="results")
    parser.add_argument("--query_store_path", default='./dataset/test_icl_query.jsonl', type=str,
                        help="results")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--index", default=[1, 2], nargs='+',
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--mutation", default=False, type=bool,
                        help="use mutation method or not")
    parser.add_argument("--candidate_num", default=40, type=int,
                        help="need be odd")
    parser.add_argument("--gpu_input", default="1", type=str)
    parser.add_argument("--icl_number", default=3, type=int)
    args = parser.parse_args()

    args.device = get_device(args)

    # if args.mutation:
    #     file_name, file_extension = os.path.splitext(args.store_path)
    #     args.store_path = f"{file_name}_muta{file_extension}"

    # 检查输出文件是否存在，若存在则删除
    output1 = args.store_path
    if os.path.exists(output1):
        os.remove(output1)

    output2 = args.query_store_path
    if os.path.exists(output2):
        os.remove(output2)

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

    # 单独保存query的内容
    with open(args.query_store_path, "w") as wf:
        for element in query_list:
            wf.write(json.dumps(element) + '\n')

    file_path0 = './dataset/label_0.jsonl'
    file_path1 = './dataset/label_1.jsonl'

    # 读取字典
    dict0 = read_jsonl_to_dict(file_path0)
    dict1 = read_jsonl_to_dict(file_path1)
    unique_keys_dict0 = {k: dict0[k] for k in dict0.keys() - dict1.keys()}
    unique_keys_dict1 = {k: dict1[k] for k in dict1.keys() - dict0.keys()}
    dict0_set = []
    for key in unique_keys_dict0:
        # input_ids = tokenizer_mlm(key, return_tensors="pt")['input_ids'].squeeze().tolist()
        dict0_set.append(key)

    dict1_set = []
    for key in unique_keys_dict1:
        # input_ids = tokenizer_mlm(key, return_tensors="pt")['input_ids'].squeeze().tolist()
        dict1_set.append(key)

    counter = 0
    for item in tqdm(eval_data):
        identifiers, code_tokens = get_identifiers(remove_comments_and_docstrings(item["func"], "c"), "c")
        processed_code = " ".join(code_tokens)

        words, sub_words, keys = _tokenize(processed_code, tokenizer_mlm)

        variable_names = []
        for name in identifiers:
            if ' ' in name[0].strip():
                continue
            variable_names.append(name[0])

        variable_substitue_dict = {}

        for tgt_word in variable_names:

            if item['target'] == 0:
                words_set = unique_keys_dict1
            else:
                words_set = unique_keys_dict0

            target_word = tgt_word

            # 找到与目标单词最接近的5个单词
            closest_words = find_closest_words(target_word, words_set, top_n=5)

            # 生成这5个单词的1到3个单词的组合
            combinations = generate_combinations(closest_words)


            for tmp_substitue in combinations:
                if tmp_substitue.strip() in variable_names:
                    continue
                if not is_valid_substitue(tmp_substitue.strip(), tgt_word, 'c'):
                    continue
                try:
                    variable_substitue_dict[tgt_word].append(tmp_substitue)
                except:
                    variable_substitue_dict[tgt_word] = [tmp_substitue]
        item["substitutes"] = variable_substitue_dict
        item["idx"] = counter % args.icl_number
        counter += 1
        with open(args.store_path, "a") as wf:
            wf.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    main()





