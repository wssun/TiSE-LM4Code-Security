import argparse
import os
import jsonlines
import random
import numpy as np
# from gensim import corpora
# from gensim.summarization import bm25
# from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel, RobertaModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq
from tqdm import tqdm
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, SequentialSampler, DataLoader
import logging
import json
import multiprocessing
import pickle
import re
import sys
sys.path.append('../../../')
sys.path.append('../../../python_parser')
# from attacker import
from python_parser.run_parser import get_identifiers, remove_comments_and_docstrings



def get_device(gpu_input):
    device = f'cuda:{gpu_input}' if torch.cuda.is_available() and int(
        gpu_input) < torch.cuda.device_count() else 'cpu'
    return device


logger = logging.getLogger(__name__)
block_size = 512
gpu_input = '0'
device = get_device(gpu_input)
data_path = {
    "Clone-detection": {
        "train_path": "../../../../ICL4code-master/dataset/Clone-detection-BigCloneBench/train_sampled.txt",
        "test_path": "../../../../ICL4code-master/dataset/Clone-detection-BigCloneBench/test_sampled.txt"
    },
    "summary": {
        "train_path": "../dataset/summary/CSN-python/python/train.jsonl",
        "test_path": "../dataset/summary/CSN-python/python/test.jsonl"
    },
    "Defect-detection": {
        "train_path": "../../../../ICL4code-master/dataset/Defect-detection/train.jsonl",
        "test_path": "../../../../ICL4code-master/dataset/Defect-detection/test.jsonl"
    },
    "Authorship-attribution": {
        "train_path": "../dataset/Authorship-attribution/train.txt",
        "test_path": "../dataset/Authorship-attribution/valid.txt"
    },
    "summary/CSN-python": {
        "train_path": "../../../../ICL4code-master/dataset/summary/CSN-python/python/train.jsonl",
        "test_path": "../../../../ICL4code-master/dataset/summary/CSN-python/python/test.jsonl"
    },
    "summary/CSN-java": {
        "train_path": "../../../../ICL4code-master/dataset/summary/CSN-java/java/train.jsonl",
        "test_path": "../../../../ICL4code-master/dataset/summary/CSN-java/java/test.jsonl"
    },
    "Code-translation": {
        "train_path": "../../../../ICL4code-master/dataset/Code-translation/train.java-cs.txt.java,../../../../ICL4code-master/dataset/Code-translation/train.java-cs.txt.cs",
        "test_path": "../../../../ICL4code-master/dataset/Code-translation/test.java-cs.txt.java,../../../../ICL4code-master/dataset/Code-translation/test.java-cs.txt.cs",
    }
}


def split_code(code):
    # 使用正则表达式匹配单词和符号
    tokens = re.findall(r"[\w']+|[.,!?;()\"']", code)
    return tokens

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def read_trans_examples(filename):
    """Read examples from filename."""
    examples=[]
    assert len(filename.split(','))==2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1,open(trg_filename) as f2:
            for line1,line2 in zip(f1,f2):
                examples.append(
                Example(
                        idx = idx,
                        source=line1.strip(),
                        target=line2.strip(),
                        )
                )
                idx+=1
    return examples


class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder

    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            outputs = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[0]
            outputs = (outputs * code_inputs.ne(1)[:, :, None]).sum(1) / code_inputs.ne(1).sum(-1)[:, None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            outputs = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[0]
            outputs = (outputs * nl_inputs.ne(1)[:, :, None]).sum(1) / nl_inputs.ne(1).sum(-1)[:, None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 url1,
                 url2

                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label
        self.url1 = url1
        self.url2 = url2


def defect_unixcoder_cocosoda_preprocess(train, test, model_path, output_path, number, args):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = RobertaModel.from_pretrained(model_path)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(model)
    model.to(device)

    code_emb = []
    model.eval()
    for obj in tqdm(train):
        with torch.no_grad():
            code_tokens = tokenizer.tokenize(' '.join(obj['code_tokens']))[:256 - 4]
            tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
            tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
            context_embeddings = model(code_inputs=torch.tensor([tokens_ids]).to(device))
            code_emb.append(context_embeddings.cpu().numpy())
    code_emb = np.concatenate(code_emb, 0)

    test_emb = []
    model.eval()
    for obj in tqdm(test):
        with torch.no_grad():
            code_tokens = tokenizer.tokenize(' '.join(obj['code_tokens']))[:256 - 4]
            tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
            tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
            context_embeddings = model(code_inputs=torch.tensor([tokens_ids]).to(device))
            test_emb.append(context_embeddings.cpu().numpy())
    test_emb = np.concatenate(test_emb, 0)

    processed = []
    for idx in tqdm(range(len(test_emb))):
        query_embeddings = test_emb[idx]
        cos_sim = F.cosine_similarity(torch.Tensor(code_emb), torch.Tensor(query_embeddings), dim=1).cpu().numpy()
        hits = []
        topk = heapq.nlargest(number, range(len(cos_sim)), cos_sim.__getitem__)
        for i in topk:
            hits.append({'score': cos_sim[i], 'corpus_id': i})

        code_candidates_tokens = []
        for i in range(len(hits)):
            code_candidates_tokens.append({'code_tokens': train[hits[i]['corpus_id']]['code_tokens'],
                                           'ori_code': train[hits[i]['corpus_id']]['ori_code'],
                                           'docstring_tokens': train[hits[i]['corpus_id']]['docstring_tokens'],
                                           'ori_docstring': train[hits[i]['corpus_id']]['ori_docstring'],
                                           'score': float(hits[i]['score']), 'idx': i + 1})
        obj = test[idx]
        processed.append({'code_tokens': obj['code_tokens'], 'docstring_tokens': obj['docstring_tokens'],
                          'code_candidates_tokens': code_candidates_tokens})
    model_name = 'unixcoder' if 'unixcoder' in model_path else 'cocosoda'
    with jsonlines.open(os.path.join(output_path, args.jsonl_path + '_' + str(number) + '.jsonl'), 'w') as f:
        f.write_all(processed)
def clone_unixcoder_cocosoda_preprocess(train, test, model_path, output_path, number, args):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = RobertaModel.from_pretrained(model_path)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(model)
    model.to(device)

    code_emb = []
    model.eval()
    for obj in tqdm(train):
        with torch.no_grad():
            code_tokens = tokenizer.tokenize(' '.join(obj['code_tokens']))[:256 - 4]
            tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
            tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
            context_embeddings = model(code_inputs=torch.tensor([tokens_ids]).to(device))
            code_emb.append(context_embeddings.cpu().numpy())
    code_emb = np.concatenate(code_emb, 0)

    test_emb = []
    model.eval()
    for obj in tqdm(test):
        with torch.no_grad():
            code_tokens = tokenizer.tokenize(' '.join(obj['code_tokens']))[:256 - 4]
            tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
            tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
            context_embeddings = model(code_inputs=torch.tensor([tokens_ids]).to(device))
            test_emb.append(context_embeddings.cpu().numpy())
    test_emb = np.concatenate(test_emb, 0)

    processed = []
    for idx in tqdm(range(len(test_emb))):
        query_embeddings = test_emb[idx]
        cos_sim = F.cosine_similarity(torch.Tensor(code_emb), torch.Tensor(query_embeddings), dim=1).cpu().numpy()
        hits = []
        topk = heapq.nlargest(number, range(len(cos_sim)), cos_sim.__getitem__)
        for i in topk:
            hits.append({'score': cos_sim[i], 'corpus_id': i})

        code_candidates_tokens = []
        for i in range(len(hits)):
            code_candidates_tokens.append({'code_tokens': train[hits[i]['corpus_id']]['code_tokens'],
                                           'ori_code': train[hits[i]['corpus_id']]['ori_code'],
                                           'docstring_tokens': train[hits[i]['corpus_id']]['docstring_tokens'],
                                           'ori_docstring': train[hits[i]['corpus_id']]['ori_docstring'],
                                           'score': float(hits[i]['score']), 'idx': i + 1,
                                           'code1': train[hits[i]['corpus_id']]['code1'],
                                           'url1': train[hits[i]['corpus_id']]['url1'],
                                           'code2': train[hits[i]['corpus_id']]['code2'],
                                           'url2': train[hits[i]['corpus_id']]['url2'],})
        obj = test[idx]
        processed.append({'code_tokens': obj['code_tokens'], 'docstring_tokens': obj['docstring_tokens'],
                          'code_candidates_tokens': code_candidates_tokens})
    model_name = 'unixcoder' if 'unixcoder' in model_path else 'cocosoda'
    with jsonlines.open(os.path.join(output_path, args.jsonl_path + '_' + str(number) + '.jsonl'), 'w') as f:
        f.write_all(processed)
def unixcoder_cocosoda_preprocess(train, test, model_path, output_path, number, args):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = RobertaModel.from_pretrained(model_path)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(model)
    model.to(device)

    code_emb = []
    model.eval()
    for obj in tqdm(train):
        with torch.no_grad():
            code_tokens = tokenizer.tokenize(' '.join(obj['code_tokens']))[:256 - 4]
            tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
            tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
            context_embeddings = model(code_inputs=torch.tensor([tokens_ids]).to(device))
            code_emb.append(context_embeddings.cpu().numpy())
    code_emb = np.concatenate(code_emb, 0)

    test_emb = []
    model.eval()
    for obj in tqdm(test):
        with torch.no_grad():
            code_tokens = tokenizer.tokenize(' '.join(obj['code_tokens']))[:256 - 4]
            tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
            tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
            context_embeddings = model(code_inputs=torch.tensor([tokens_ids]).to(device))
            test_emb.append(context_embeddings.cpu().numpy())
    test_emb = np.concatenate(test_emb, 0)

    processed = []
    for idx in tqdm(range(len(test_emb))):
        query_embeddings = test_emb[idx]
        cos_sim = F.cosine_similarity(torch.Tensor(code_emb), torch.Tensor(query_embeddings), dim=1).cpu().numpy()
        hits = []
        topk = heapq.nlargest(number, range(len(cos_sim)), cos_sim.__getitem__)
        for i in topk:
            hits.append({'score': cos_sim[i], 'corpus_id': i})

        code_candidates_tokens = []
        for i in range(len(hits)):
            code_candidates_tokens.append({'code_tokens': train[hits[i]['corpus_id']]['code_tokens'],
                                           'docstring_tokens': train[hits[i]['corpus_id']]['docstring_tokens'],
                                           'score': float(hits[i]['score']), 'idx': i + 1})
        obj = test[idx]
        processed.append({'code_tokens': obj['code_tokens'], 'docstring_tokens': obj['docstring_tokens'],
                          'code_candidates_tokens': code_candidates_tokens})
    model_name = 'unixcoder' if 'unixcoder' in model_path else 'cocosoda'
    with jsonlines.open(os.path.join(output_path, model_name + '_' + str(number) + '.jsonl'), 'w') as f:
        f.write_all(processed)

def summary_unixcoder_cocosoda_preprocess(train, test, model_path, output_path, number, args):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = RobertaModel.from_pretrained(model_path)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(model)
    model.to(device)

    code_emb = []
    model.eval()
    for obj in tqdm(train):
        with torch.no_grad():
            code_tokens = tokenizer.tokenize(' '.join(obj['code_tokens']))[:256 - 4]
            tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
            tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
            context_embeddings = model(code_inputs=torch.tensor([tokens_ids]).to(device))
            code_emb.append(context_embeddings.cpu().numpy())
    code_emb = np.concatenate(code_emb, 0)

    test_emb = []
    model.eval()
    for obj in tqdm(test):
        with torch.no_grad():
            code_tokens = tokenizer.tokenize(' '.join(obj['code_tokens']))[:256 - 4]
            tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
            tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
            context_embeddings = model(code_inputs=torch.tensor([tokens_ids]).to(device))
            test_emb.append(context_embeddings.cpu().numpy())
    test_emb = np.concatenate(test_emb, 0)

    processed = []
    for idx in tqdm(range(len(test_emb))):
        query_embeddings = test_emb[idx]
        cos_sim = F.cosine_similarity(torch.Tensor(code_emb), torch.Tensor(query_embeddings), dim=1).cpu().numpy()
        hits = []
        topk = heapq.nlargest(number, range(len(cos_sim)), cos_sim.__getitem__)
        for i in topk:
            hits.append({'score': cos_sim[i], 'corpus_id': i})

        code_candidates_tokens = []
        for i in range(len(hits)):
            code_candidates_tokens.append({'code_tokens': train[hits[i]['corpus_id']]['code_tokens'],
                                           'ori_code': train[hits[i]['corpus_id']]['code'],
                                           'docstring_tokens': train[hits[i]['corpus_id']]['docstring_tokens'],
                                           'ori_docstring': train[hits[i]['corpus_id']]['docstring'],
                                           'score': float(hits[i]['score']), 'idx': i + 1})
        obj = test[idx]
        processed.append({'code_tokens': obj['code_tokens'], 'ori_code': obj['code'], 'docstring_tokens': obj['docstring_tokens'],
                          'code_candidates_tokens': code_candidates_tokens})
    model_name = 'unixcoder' if 'unixcoder' in model_path else 'cocosoda'
    with jsonlines.open(os.path.join(output_path, args.jsonl_path + '_' + str(number) + '.jsonl'), 'w') as f:
        f.write_all(processed)

def trans_unixcoder_cocosoda_preprocess(train, test, model_path, output_path, number, args):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = RobertaModel.from_pretrained(model_path)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(model)
    model.to(device)

    code_emb = []
    model.eval()
    for obj in tqdm(train):
        with torch.no_grad():
            code_tokens = tokenizer.tokenize(' '.join(obj['code_tokens']))[:256 - 4]
            tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
            tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
            context_embeddings = model(code_inputs=torch.tensor([tokens_ids]).to(device))
            code_emb.append(context_embeddings.cpu().numpy())
    code_emb = np.concatenate(code_emb, 0)

    test_emb = []
    model.eval()
    for obj in tqdm(test):
        with torch.no_grad():
            code_tokens = tokenizer.tokenize(' '.join(obj['code_tokens']))[:256 - 4]
            tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
            tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
            context_embeddings = model(code_inputs=torch.tensor([tokens_ids]).to(device))
            test_emb.append(context_embeddings.cpu().numpy())
    test_emb = np.concatenate(test_emb, 0)

    processed = []
    for idx in tqdm(range(len(test_emb))):
        query_embeddings = test_emb[idx]
        cos_sim = F.cosine_similarity(torch.Tensor(code_emb), torch.Tensor(query_embeddings), dim=1).cpu().numpy()
        hits = []
        topk = heapq.nlargest(number, range(len(cos_sim)), cos_sim.__getitem__)
        for i in topk:
            hits.append({'score': cos_sim[i], 'corpus_id': i})

        code_candidates_tokens = []
        for i in range(len(hits)):
            code_candidates_tokens.append({'code_tokens': train[hits[i]['corpus_id']]['code_tokens'],
                                           'ori_code': train[hits[i]['corpus_id']]['code'],
                                           'docstring_tokens': train[hits[i]['corpus_id']]['docstring_tokens'],
                                           'ori_docstring': train[hits[i]['corpus_id']]['docstring'],
                                           'score': float(hits[i]['score']), 'idx': i + 1})
        obj = test[idx]
        processed.append({'code_tokens': obj['code_tokens'], 'ori_code': obj['code'], 'docstring_tokens': obj['docstring_tokens'],
                          'ori_docstring': obj['docstring'],
                          'code_candidates_tokens': code_candidates_tokens})
    model_name = 'unixcoder' if 'unixcoder' in model_path else 'cocosoda'
    with jsonlines.open(os.path.join(output_path, args.jsonl_path + '_' + str(number) + '.jsonl'), 'w') as f:
        f.write_all(processed)

def CloneText(file_path, block_size=512):
    index_filename = file_path
    url_to_code = {}
    # 读取了所有的数据集文件.
    with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl') as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            url_to_code[js['idx']] = js['func']
            # idx 表示每段代码的id
    data = []
    with open(index_filename) as f:
        for line in tqdm(f):
            cache = {}
            line = line.strip()
            url1, url2, ori_label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                # 在data.jsonl中不存在，直接跳过
                continue
            label = 'Yes' if ori_label == '1' else 'No'
            code_token = url_to_code[url1] + url_to_code[url2]
            __, tokens = get_identifiers(remove_comments_and_docstrings(code_token, "java"), "java")
            if len(tokens) > 1000:
                # 如果代码分词结果超过，不加入到data中
                continue
            cache['ori_code'] = code_token
            cache['code_tokens'] = tokens
            cache['docstring_tokens'] = label
            cache['ori_docstring'] = ori_label
            cache['url1'] = url1
            cache['url2'] = url2
            cache['code1'] = url_to_code[url1]
            cache['code2'] = url_to_code[url2]
            data.append(cache)
    return data


def data_load(train_path, test_path):
    if 'summary' in train_path:
        train = []
        with jsonlines.open(train_path) as f:
            for i in f:
                if len(i['code_tokens']) > 1000: continue
                train.append(i)
        test = []
        with jsonlines.open(test_path) as f:
            for i in f:
                if len(i['code_tokens']) > 1000: continue
                test.append(i)

    elif 'Clone-detection' in train_path:
        test = CloneText(file_path=test_path, block_size=block_size)
        train = CloneText(file_path=train_path, block_size=block_size)

    elif 'Defect-detection' in train_path:
        train = []
        with jsonlines.open(train_path) as f:
            for i in tqdm(f):
                data = {}
                __, tokens = get_identifiers(remove_comments_and_docstrings(i['func'], "c"), "c")
                if len(tokens) > 1000:
                    # 如果代码分词结果超过，不加入到data中
                    continue
                data['code_tokens'] = tokens
                data['ori_code'] = i['func']
                label = 'Yes' if i['target'] == 1 else 'No'
                data['docstring_tokens'] = label
                data['ori_docstring'] = i['target']
                train.append(data)
        test = []
        with jsonlines.open(test_path) as f:
            for i in tqdm(f):
                data = {}
                __, tokens = get_identifiers(remove_comments_and_docstrings(i['func'], "c"), "c")
                if len(tokens) > 1000:
                    # 如果代码分词结果超过，不加入到data中
                    continue
                data['code_tokens'] = tokens
                data['ori_code'] = i['func']
                label = 'Yes' if i['target'] == 1 else 'No'
                data['docstring_tokens'] = label
                data['ori_docstring'] = i['target']
                test.append(data)

    elif 'Authorship-attribution' in train_path:
        train = []
        with open(train_path) as f:
            for line in f:
                data = {}
                code = line.split(" <CODESPLIT> ")[0]
                code = code.replace("\\n", "\n").replace('\"', '"')
                label_ori = line.split(" <CODESPLIT> ")[1]
                label = re.sub(r'\n', '', label_ori)
                data['code_tokens'] = split_code(code)
                data['docstring_tokens'] = str(label)
                train.append(data)
        test = []
        with open(test_path) as f:
            for line in f:
                data = {}
                code = line.split(" <CODESPLIT> ")[0]
                code = code.replace("\\n", "\n").replace('\"', '"')
                label_ori = line.split(" <CODESPLIT> ")[1]
                label = re.sub(r'\n', '', label_ori)
                data['code_tokens'] = split_code(code)
                data['docstring_tokens'] = str(label)
                test.append(data)

    elif 'Code-translation' in train_path:
        train = []
        filename = train_path
        assert len(filename.split(',')) == 2
        src_filename = filename.split(',')[0]
        trg_filename = filename.split(',')[1]

        with open(src_filename) as f1, open(trg_filename) as f2:
            for line1, line2 in zip(f1, f2):
                data = {}
                data['code'] = line1
                __, java_tokens = get_identifiers(remove_comments_and_docstrings(line1, "java"), "java")
                data['code_tokens'] = java_tokens
                data['docstring'] = line2
                __, cs_tokens = get_identifiers(remove_comments_and_docstrings(line2, "c"), "c")
                data['docstring_tokens'] = cs_tokens
                train.append(data)
        test = []
        filename = test_path
        assert len(filename.split(',')) == 2
        src_filename = filename.split(',')[0]
        trg_filename = filename.split(',')[1]

        with open(src_filename) as f1, open(trg_filename) as f2:
            for line1, line2 in zip(f1, f2):
                data = {}
                data['code'] = line1
                __, java_tokens = get_identifiers(remove_comments_and_docstrings(line1, "java"), "java")
                data['code_tokens'] = java_tokens
                data['docstring'] = line2
                __, cs_tokens = get_identifiers(remove_comments_and_docstrings(line2, "c"), "c")
                data['docstring_tokens'] = cs_tokens
                test.append(data)


    return train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='code_translation', type=str)
    parser.add_argument("--icl_number", default=7, type=int)
    parser.add_argument("--small_case", default=False, type=bool)
    parser.add_argument("--jsonl_path", default='test_222r', type=str,
                        help="results")
    args = parser.parse_args()

    if 'defect' in args.task:
        task = 'Defect-detection'
    elif 'clone' in args.task:
        task = 'Clone-detection'
    elif 'adv_summary_java' == args.task:
        task = 'summary/CSN-java'
    elif 'adv_summary_python' == args.task:
        task = 'summary/CSN-python'
    elif 'code_translation' == args.task:
        task = 'Code-translation'

    train, test = data_load(data_path[task]['train_path'], data_path[task]['test_path'])

    if args.small_case:
        # train = train[:100]
        # test = test[:100]
        train = train
        test = test[:500]

    if ('defect' or 'clone' or 'summary_java' or 'summary_python') == args.task:
        unixcoder_cocosoda_preprocess(train, test, '../../../../ICL4code-master/unixcoder-base', './dataset/', args.icl_number, args)
    elif 'adv_defect' == args.task:
        defect_unixcoder_cocosoda_preprocess(train, test, '../../../../ICL4code-master/unixcoder-base', './dataset/', args.icl_number, args)
    elif 'adv_clone' == args.task:
        clone_unixcoder_cocosoda_preprocess(train, test, '../unixcoder-base', './dataset/', args.icl_number, args)
    elif ('adv_summary_java' == args.task) or ('adv_summary_python' == args.task):
        summary_unixcoder_cocosoda_preprocess(train, test, '../../../../ICL4code-master/unixcoder-base', './dataset/', args.icl_number, args)
    elif 'code_translation' == args.task:
        trans_unixcoder_cocosoda_preprocess(train, test, '../../../../ICL4code-master/unixcoder-base', './dataset/', args.icl_number, args)



if __name__ == '__main__':
    main()

