import json
import sys
import copy
import torch
import argparse
from sklearn.cluster import KMeans
from tqdm import tqdm

sys.path.append('../../../')
sys.path.append('../../../python_parser')

# from attacker import 
from python_parser.run_parser import get_identifiers, remove_comments_and_docstrings
from utils import is_valid_variable_name, _tokenize, get_identifier_posistions_from_code, get_masked_code_by_position, get_substitues, is_valid_substitue
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_data_file", default='./dataset/test.jsonl', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--base_model", default='../../../codebert-base-mlm', type=str,
                        help="Base Model")
    parser.add_argument("--store_path", default='./dataset/test_subs.jsonl', type=str,
                        help="results")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--index", default=[0, 2], nargs='+',
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--mutation", default=True, type=bool,
                        help="use mutation method or not")
    args = parser.parse_args()

    eval_data = []

    codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)
    codebert_mlm.to('cuda')

    with open(args.eval_data_file) as rf:
        for i, line in enumerate(rf):
            if i < int(args.index[0]) or i >= int(args.index[1]):
                continue
            item = json.loads(line.strip())
            eval_data.append(item)
    print(len(eval_data))
    with open(args.store_path, "w") as wf:
        for item in tqdm(eval_data):
            identifiers, code_tokens = get_identifiers(remove_comments_and_docstrings(item["func"], "c"), "c")
            processed_code = " ".join(code_tokens)

            words, sub_words, keys = _tokenize(processed_code, tokenizer_mlm)

            variable_names = []
            for name in identifiers:
                if ' ' in name[0].strip():
                    continue
                variable_names.append(name[0])

            sub_words = [tokenizer_mlm.cls_token] + sub_words[:args.block_size - 2] + [tokenizer_mlm.sep_token]

            input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])

            word_predictions = codebert_mlm(input_ids_.to('cuda'))[
                0].squeeze()  # seq-len(sub) vocab （seq:序列长度； len：词表大小）
            word_pred_scores_all, word_predictions = torch.topk(word_predictions, 60, -1)  # seq-len k
            # 得到前k个结果.

            word_predictions = word_predictions[1:len(sub_words) + 1, :]
            word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]

            names_positions_dict = get_identifier_posistions_from_code(words, variable_names)

            variable_substitue_dict = {}
            with torch.no_grad():
                orig_embeddings = codebert_mlm.roberta(input_ids_.to('cuda'))[0]

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
                    # 找到要替换的变量所在位置填空预测的结果
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

                        with torch.no_grad():
                            new_embeddings = codebert_mlm.roberta(new_ids_.to('cuda'))[0]
                        # 比较新\旧变量的cos，利用不含lm_head的原模型预测的embedding结果
                        new_word_embed = new_embeddings[0][keys[one_pos][0] + 1:keys[one_pos][1] + 1]

                        sims.append((i, sum(cos(orig_word_embed, new_word_embed)) / subwords_leng))
                        embed_vectors.append(new_word_embed.cpu().numpy().flatten())

                    if args.mutation:
                        sims = []
                        embed_vectors = torch.tensor(embed_vectors)
                        # Apply KMeans clustering
                        n_clusters = 10
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        clusters = kmeans.fit_predict(embed_vectors)
                        for cluster_id in range(n_clusters):
                            # Get indices of samples belonging to the current cluster
                            cluster_indices = [i for i, x in enumerate(clusters) if x == cluster_id]
                            # Extract embeddings of samples in the current cluster
                            cluster_embeds = embed_vectors[cluster_indices].to('cuda')
                            # Calculate cosine similarities between the original word embedding and all samples in the cluster
                            similarities = cos(orig_word_embed.flatten(), cluster_embeds)
                            # Find the index of the most similar sample within the cluster
                            most_similar_idx = torch.argmax(similarities)
                            # most_similar_substitute = substitutes[:, cluster_indices[most_similar_idx]].reshape(-1, 1)
                            # most_similar_score = word_pred_scores[:, cluster_indices[most_similar_idx]].reshape(-1, 1)
                            sims.append((cluster_indices[most_similar_idx], similarities[most_similar_idx]))

                    sims = sorted(sims, key=lambda x: x[1], reverse=True)

                    sims_range = range(n_clusters) if args.mutation else range(int(nums_candis / 2))
                    for i in sims_range:
                        # 从60个候选中选30
                        similar_substitutes.append(substitutes[:, sims[i][0]].reshape(subwords_leng, -1))
                        similar_word_pred_scores.append(word_pred_scores[:, sims[i][0]].reshape(subwords_leng, -1))

                    similar_substitutes = torch.cat(similar_substitutes, 1)
                    similar_word_pred_scores = torch.cat(similar_word_pred_scores, 1)

                    substitutes = get_substitues(similar_substitutes,
                                                 tokenizer_mlm,
                                                 codebert_mlm,
                                                 1,
                                                 similar_word_pred_scores,
                                                 0)
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
            wf.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    main()
