import json
import sys
import copy
import torch
import argparse
from tqdm import tqdm
import os
from sklearn.cluster import KMeans
import re

sys.path.append('../../../')
sys.path.append('../../../python_parser')

# from attacker import
from python_parser.run_parser import get_identifiers, remove_comments_and_docstrings
from utils import is_valid_variable_name, _tokenize, get_identifier_posistions_from_code, get_masked_code_by_position, \
     get_substitues, is_valid_substitue
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


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

    file_path0 = results_filepath + '_label_0.jsonl'
    file_path1 = results_filepath + '_label_1.jsonl'

    # file_path0 = './dataset/label_0.jsonl'
    # file_path1 = './dataset/label_1.jsonl'
    # 读取字典
    dict0 = read_jsonl_to_dict(file_path0)
    dict1 = read_jsonl_to_dict(file_path1)
    dict0_set = set()
    for key in dict0:
        input_ids = tokenizer_mlm(key, return_tensors="pt")['input_ids'].squeeze().tolist()
        dict0_set.update(input_ids)

    dict1_set = set()
    for key in dict1:
        input_ids = tokenizer_mlm(key, return_tensors="pt")['input_ids'].squeeze().tolist()
        dict1_set.update(input_ids)

    # 准备mask，该遮罩只允许另一类中的 ids
    vocab_size = tokenizer_mlm.vocab_size
    mask0 = torch.ones(vocab_size, dtype=torch.float32).to(args.device)
    mask0[list(dict0_set)] = 0.0
    mask1 = torch.ones(vocab_size, dtype=torch.float32).to(args.device)
    mask1[list(dict1_set)] = 0.0

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

        # 似乎codellama生成embedding不可以加特殊token（bos，eos等）？？
        # sub_words = [tokenizer_mlm.bos_token] + sub_words[:args.block_size - 2] + [tokenizer_mlm.eos_token]
        sub_words = sub_words[:args.block_size]

        input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])
        # input_ids_ = tokenizer_mlm(sub_words, return_tensors='pt', add_special_tokens=False)

        word_predictions = codebert_mlm(input_ids_.to(args.device))[0].squeeze()  # seq-len(sub) vocab

        # 应用遮罩
        if item['target'] == 0:
            word_predictions = word_predictions * mask0
        else:
            word_predictions = word_predictions * mask1

        # 得到前k个结果.
        word_pred_scores_all, word_predictions = torch.topk(word_predictions, args.candidate_num, -1)  # seq-len k


        word_predictions = word_predictions[1:len(sub_words) + 1, :]
        word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]

        names_positions_dict = get_identifier_posistions_from_code(words, variable_names)

        variable_substitue_dict = {}
        with torch.no_grad():
            orig_embeddings = codebert_mlm.model(input_ids_.to(args.device))[0]

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        for tgt_word in names_positions_dict.keys():
            tgt_positions = names_positions_dict[tgt_word]  # the positions of tgt_word in code
            if not is_valid_variable_name(tgt_word, lang='c'):
                # if the extracted name is not valid
                continue

                ## 得到(所有位置的)substitues
            all_substitues = []
            for one_pos in tgt_positions:
                ## 一个变量名会出现很多次
                if keys[one_pos][0] >= word_predictions.size()[0]:
                    continue
                substitutes = word_predictions[keys[one_pos][0]:keys[one_pos][1]]  # L, k
                word_pred_scores = word_pred_scores_all[keys[one_pos][0]:keys[one_pos][1]]

                orig_word_embed = orig_embeddings[0][keys[one_pos][0] + 1:keys[one_pos][1] + 1]

                similar_substitutes = []
                similar_word_pred_scores = []
                sims = []
                embed_vectors = []
                subwords_leng, nums_candis = substitutes.size()

                for i in range(nums_candis):
                    new_ids_ = copy.deepcopy(input_ids_)
                    new_ids_[0][keys[one_pos][0] + 1:keys[one_pos][1] + 1] = substitutes[:, i]
                    # 替换词得到新embeddings
                    # codellama的sim平均似乎小于codebert。计算平均值？（词表更小引起的？）
                    with torch.no_grad():
                        new_embeddings = codebert_mlm.model(new_ids_.to(args.device))[0]
                    new_word_embed = new_embeddings[0][keys[one_pos][0] + 1:keys[one_pos][1] + 1]
                    sims.append((i, sum(cos(orig_word_embed, new_word_embed)) / subwords_leng))
                    embed_vectors.append(new_word_embed.cpu().numpy().flatten())


                # mutation method (k-means)
                if args.mutation:
                    sims = []
                    embed_vectors = torch.tensor(embed_vectors)
                    # Apply KMeans clustering
                    n_clusters = round(args.candidate_num/6)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(embed_vectors)
                    for cluster_id in range(n_clusters):
                        # Get indices of samples belonging to the current cluster
                        cluster_indices = [i for i, x in enumerate(clusters) if x == cluster_id]
                        # Extract embeddings of samples in the current cluster
                        cluster_embeds = embed_vectors[cluster_indices].to(args.device)
                        # Calculate cosine similarities between the original word embedding and all samples in the cluster
                        similarities = cos(orig_word_embed.flatten(), cluster_embeds)
                        # Find the index of the most similar sample within the cluster
                        most_similar_idx = torch.argmax(similarities)
                        # most_similar_substitute = substitutes[:, cluster_indices[most_similar_idx]].reshape(-1, 1)
                        # most_similar_score = word_pred_scores[:, cluster_indices[most_similar_idx]].reshape(-1, 1)
                        sims.append((cluster_indices[most_similar_idx], similarities[most_similar_idx]))

                sims = sorted(sims, key=lambda x: x[1], reverse=True)
                select_range = range(n_clusters) if args.mutation else range(int(nums_candis / 2))
                for i in select_range:
                    similar_substitutes.append(substitutes[:, sims[i][0]].reshape(subwords_leng, -1))
                    similar_word_pred_scores.append(word_pred_scores[:, sims[i][0]].reshape(subwords_leng, -1))

                similar_substitutes = torch.cat(similar_substitutes, 1)
                similar_word_pred_scores = torch.cat(similar_word_pred_scores, 1)

                substitutes = get_substitues(similar_substitutes,
                                             tokenizer_mlm,
                                             codebert_mlm,
                                             1,
                                             similar_word_pred_scores,
                                             0,
                                             args.device)
                all_substitues += substitutes
            all_substitues = set(all_substitues)

            for tmp_substitue in all_substitues:
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
