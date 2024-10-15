# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np

system_prompt = "You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
prompt_front = 'You are given a code function:\n'
prompt_back = '\nIs the following function has any potential vulnerability? Please answer ‘Yes’ or ‘No’. '


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.device = args.device
        self.query = 0

    def forward(self, input_ids=None, labels=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        logits = outputs
        prob = F.sigmoid(logits)
        # print(prob.size())
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob

    def get_results(self, dataset, batch_size):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=4,
                                     pin_memory=False)

        self.eval()
        logits = []
        labels = []
        for batch in eval_dataloader:
            inputs = batch[0].to(self.device)
            label = batch[1].to(self.device)
            with torch.no_grad():
                lm_loss, logit = self.forward(inputs, label)
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())
        # print(logits.size())

        logits = np.concatenate(logits, 0)
        labels = np.concatenate(labels, 0)

        probs = [[1 - prob[0], prob[0]] for prob in logits]
        pred_labels = [1 if label else 0 for label in logits[:, 0] > 0.5]

        return probs, pred_labels


class CodeLlama(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeLlama, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.query = 0
        self.device = args.device
        self.tokenizer.pad_token = tokenizer.eos_token

    def forward(self, input_ids=None, labels=None):
        # 输出最大 token 数
        max_token = 4

        # 将 system 和 prompt 转换成 tokens
        sys = f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n'
        task_prompt_back = f'{prompt_back} [/INST]'
        sys_tokens = self.tokenizer.tokenize(sys)
        task_prompt_back_tokens = self.tokenizer.tokenize(task_prompt_back)
        task_prompt_front_tokens = self.tokenizer.tokenize(prompt_front)

        # 将 sys 和 prompt 转换成 tensor
        sys_tensor = torch.tensor(self.tokenizer.convert_tokens_to_ids(sys_tokens)).view(1, -1).to(self.device)
        task_prompt_back_tensor = torch.tensor(self.tokenizer.convert_tokens_to_ids(task_prompt_back_tokens)).view(1,
                                                                                                                   -1).to(
            self.device)
        task_prompt_front_tensor = torch.tensor(self.tokenizer.convert_tokens_to_ids(task_prompt_front_tokens)).view(1,
                                                                                                                     -1).to(
            self.device)

        # 重复 sys 和 prompt 以匹配 input_ids 的批次大小
        batch_size = input_ids.shape[0]
        sys_tensor = sys_tensor.repeat(batch_size, 1)
        task_prompt_back_tensor = task_prompt_back_tensor.repeat(batch_size, 1)
        task_prompt_front_tensor = task_prompt_front_tensor.repeat(batch_size, 1)

        # 将 sys 和 prompt 拼接到 input_ids 的前后
        input_ids = torch.cat((sys_tensor, task_prompt_front_tensor, input_ids, task_prompt_back_tensor), dim=1)

        # 使用 generate 进行生成
        outputs = self.encoder.generate(input_ids, max_new_tokens=max_token, output_scores=True,
                                        return_dict_in_generate=True, pad_token_id=self.tokenizer.pad_token_id)

        logits = outputs.scores
        probabilities = torch.zeros(batch_size, 1).to(self.device)

        for i in range(max_token):
            raw_logits = logits[i]
            max_idx = torch.argmax(raw_logits, dim=1)
            # print(max_idx.shape, max_idx[:3])
            # import sys
            # sys.exit(-1)
            # 判断 max_idx 是否属于 no_token_id 或 yes_token_id
            mask = (max_idx == self.args.no_token_id) | (max_idx == self.args.yes_token_id)

            if not mask.any():
                continue

            left_logits = raw_logits[:, self.args.no_token_id]
            right_logits = raw_logits[:, self.args.yes_token_id]

            # 将左侧和右侧的 logits 拼接
            combined_logits = torch.cat((left_logits.unsqueeze(1), right_logits.unsqueeze(1)), dim=1)

            # 计算 softmax 概率
            prob = F.softmax(combined_logits, dim=1)

            # 将 "yes" 概率提取出来
            yes_prob = prob[:, 1]

            # 更新 probabilities
            probabilities[mask] = yes_prob[mask].view(-1, 1)

        return probabilities

    def icl_forward(self, chat_history=None, query=None):
        # 输出最大 token 数
        max_token = 4

        # 将 system 和 prompt 转换成 tokens
        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
        do_strip = False
        for user_input, response in chat_history:
            user_input = user_input.strip() if do_strip else user_input
            do_strip = True
            texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
        message = query.strip() if do_strip else query
        texts.append(f'{message} [/INST]')

        xx = ' '.join(texts)
        mes_tokens = self.tokenizer.tokenize(xx)
        # 将 sys 和 prompt 转换成 tensor
        mes_tensor = torch.tensor(self.tokenizer.convert_tokens_to_ids(mes_tokens)).view(1, -1).to(self.device)



        # 使用 generate 进行生成
        outputs = self.encoder.generate(mes_tensor, max_new_tokens=max_token, output_scores=True,
                                        return_dict_in_generate=True, pad_token_id=self.tokenizer.pad_token_id)

        logits = outputs.scores
        probabilities = torch.zeros(1, 1).to(self.device)

        for i in range(max_token):
            raw_logits = logits[i]
            max_idx = torch.argmax(raw_logits, dim=1)
            # print(max_idx.shape, max_idx[:3])
            # import sys
            # sys.exit(-1)
            # 判断 max_idx 是否属于 no_token_id 或 yes_token_id
            mask = (max_idx == self.args.no_token_id) | (max_idx == self.args.yes_token_id)

            if not mask.any():
                continue

            left_logits = raw_logits[:, self.args.no_token_id]
            right_logits = raw_logits[:, self.args.yes_token_id]

            # 将左侧和右侧的 logits 拼接
            combined_logits = torch.cat((left_logits.unsqueeze(1), right_logits.unsqueeze(1)), dim=1)

            # 计算 softmax 概率
            prob = F.softmax(combined_logits, dim=1)

            # 将 "yes" 概率提取出来
            yes_prob = prob[:, 1]

            # 更新 probabilities
            probabilities[mask] = yes_prob[mask].view(-1, 1)

        return probabilities

    def forward_demo(self, input_ids=None, labels=None):
        # 输出最大 token 数
        max_token = 4

        # 将 system 和 prompt 转换成 tokens
        sys = f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n'
        task_prompt = f'\n{prompt} [/INST]'
        sys_tokens = self.tokenizer.tokenize(sys)
        task_prompt_tokens = self.tokenizer.tokenize(task_prompt)

        # 将 sys 和 prompt 转换成 tensor
        sys_tensor = torch.tensor(self.tokenizer.convert_tokens_to_ids(sys_tokens)).view(1, -1).to(self.device)
        task_prompt_tensor = torch.tensor(self.tokenizer.convert_tokens_to_ids(task_prompt_tokens)).view(1, -1).to(
            self.device)

        # 重复 sys 和 prompt 以匹配 input_ids 的批次大小
        batch_size = input_ids.shape[0]
        sys_tensor = sys_tensor.repeat(batch_size, 1)
        task_prompt_tensor = task_prompt_tensor.repeat(batch_size, 1)

        # 将 sys 和 prompt 拼接到 input_ids 的前后
        input_ids = torch.cat((sys_tensor, input_ids, task_prompt_tensor), dim=1)

        # 使用 generate 进行生成
        output = self.encoder(input_ids)
        logits = output[0]
        prob = torch.softmax(logits, dim=1).to(self.args.device)
        # logits = torch.softmax(logits, dim=1).to(self.args.device)

        return prob

    def get_results(self, dataset, batch_size):
        '''Given a dataset, return probabilities and labels.'''
        #  每次计算important-score都统计为一次query
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0,
                                     pin_memory=False)

        self.eval()
        logits = []
        for batch in eval_dataloader:
            inputs = batch[0].to(self.device)
            label = batch[1].to(self.device)
            with torch.no_grad():
                logit = self.forward(inputs, label)
                logits.append(logit.cpu().numpy())

        logits = np.concatenate(logits, 0)

        probs = [[1 - prob[0], prob[0]] for prob in logits]
        pred_labels = [1 if label else 0 for label in logits[:, 0] > 0.5]

        return probs, pred_labels

    def get_icl_results(self, dataset, query, icl_num):
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=icl_num, num_workers=0, pin_memory=False)
        self.eval()
        logits = []

        for batch in eval_dataloader:
            inputs = batch[0]
            labels = batch[1]
            demo_history = []
            with torch.no_grad():
                for i in range(icl_num):
                    demo_history.append((prompt_front + inputs[i] + prompt_back, labels[i]))
                logit = self.icl_forward(demo_history, prompt_front + query + prompt_back)
                logits.append(logit.cpu().numpy())

        logits = np.concatenate(logits, 0)
        probs = [[1 - prob[0], prob[0]] for prob in logits]
        pred_labels = [1 if label else 0 for label in logits[:, 0] > 0.5]

        return probs, pred_labels

    def get_results_singleB(self, dataset, batch_size):
        '''Given a dataset, return probabilities and labels.'''
        #  每次计算important-score都统计为一次query
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=1, num_workers=0, pin_memory=False)

        self.eval()
        logits = []
        for batch in eval_dataloader:
            inputs = batch[0].to(self.device)
            label = batch[1].to(self.device)
            with torch.no_grad():
                logit = self.forward(inputs, label)
                logits.append(logit.cpu().numpy())

        logits = np.concatenate(logits, 0)

        probs = [[1 - prob[0], prob[0]] for prob in logits]
        pred_labels = [1 if label else 0 for label in logits[:, 0] > 0.5]

        return probs, pred_labels
