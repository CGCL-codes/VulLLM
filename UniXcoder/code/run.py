# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import pandas as pd
import random
import json
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import numpy as np
import pickle

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
import tqdm
from tqdm import tqdm

from model import *
from torch.optim import AdamW
from transformers import (get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
import sys

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 index,
                 label,

                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.index = index
        self.label = label


def convert_examples_to_features(js, tokenizer, args):
    """convert examples to token ids"""
    code = js[args.code_key]
    if args.tokenize_ast_token:
        code = ' '.join(js['ast_tokens'])
    code = ' '.join(code.split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 4]
    source_tokens = [tokenizer.cls_token, "<encoder_only>", tokenizer.sep_token] + code_tokens + [
        tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, 0, int(js[args.label_key]))


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        m = re.search(r"(train|valid|test)", file_path)
        if m is None:
            partition = None
        else:
            partition = m.group(1)

        self.examples = []
        data = []
        count = 0
        with open(file_path) as f:
            datas = json.load(f)
            for js in datas:
                data.append(js)
                count += 1
                if args.head is not None and count >= args.head:
                    break
        for js in data:
            self.examples.append(convert_examples_to_features(js, tokenizer, args))

        if partition == 'train':
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (
            torch.tensor(self.examples[i].input_ids),
            torch.tensor(self.examples[i].label),
        )


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=4, pin_memory=True)

    args.max_steps = args.num_train_epochs * len(train_dataloader)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * 0.1,
                                                num_training_steps=args.max_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // args.n_gpu)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Effective train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    losses, best_perf = [], 0

    model.zero_grad()
    for idx in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            model.train()
            loss, prob = model(input_ids=inputs, labels=labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            losses.append(loss.item())

            if (step + 1) % 100 == 0:
                logger.info("epoch {} step {} loss {}".format(idx, step + 1, round(np.mean(losses[-100:]), 4)))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        results = evaluate(args, model, tokenizer, args.eval_data_file)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value, 4))

        if results[f'eval_{args.validation_metric}'] > best_perf:
            best_perf = results[f'eval_{args.validation_metric}']
            logger.info("  " + "*" * 20)
            logger.info(f"  Best {args.validation_metric}:%s", round(best_perf, 4))
            logger.info("  " + "*" * 20)

            checkpoint_prefix = 'checkpoint-best-f1'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer, data_file):
    """ Evaluate the model """
    eval_dataset = TextDataset(tokenizer, args, data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    logger.info("  Gradient accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Effective batch size = %d", args.eval_batch_size * args.gradient_accumulation_steps)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    labels = []
    probs = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            loss, prob = model(input_ids=inputs, labels=label)
            eval_loss += loss.mean().item()
            labels.append(label.cpu().numpy())
            probs.append(prob.cpu().numpy())
        nb_eval_steps += 1
    labels = np.concatenate(labels, 0)
    probs = np.concatenate(probs, 0)
    preds = np.argmax(probs, axis=1)
    eval_loss = eval_loss / nb_eval_steps

    result = {
        "eval_acc": accuracy_score(labels, preds),
        "eval_precision": precision_score(labels, preds),
        "eval_recall": recall_score(labels, preds),
        "eval_f1": f1_score(labels, preds),
    }

    return result


def test(args, model, tokenizer, data_file):
    """ Evaluate the model """
    eval_dataset = TextDataset(tokenizer, args, data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    logger.info("  Gradient accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Effective batch size = %d", args.eval_batch_size * args.gradient_accumulation_steps)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    labels = []
    probs = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            loss, prob = model(input_ids=inputs, labels=label)
            eval_loss += loss.mean().item()
            labels.append(label.cpu().numpy())
            probs.append(prob.cpu().numpy())

        nb_eval_steps += 1
    labels = np.concatenate(labels, 0)
    probs = np.concatenate(probs, 0)
    preds = np.argmax(probs, axis=1)

    eval_loss = eval_loss / nb_eval_steps
    print(labels[:20])
    print(preds[:20])

    with open(data_file, 'r') as json_file:
        data = json.load(json_file)
        cwe_list = [item['CWE'] for item in data]  # 假设每个项目都有一个 'CWE' 键

    # 确保 CWE 列表的长度与其他数据相匹配
    if len(cwe_list) != len(labels):
        raise ValueError("CWE 列表的长度与标签的长度不匹配")

    if not os.path.exists(args.csv_path):
        with open(args.csv_path, 'w') as f:
            f.write('CWE,Label,Prediction,Prob\n')

    probs2 = [[prob[0], 1 - prob[0]] for prob in probs]
    for cwe, label, pred, prob in zip(cwe_list, labels, preds, probs2):
        temp_df = pd.DataFrame({'CWE': [cwe], 'Label': [label], 'Prediction': [pred], 'Prob': [prob]})
        temp_df.to_csv(args.csv_path, index=False, mode='a', header=False)
    result = {
        "eval_acc": accuracy_score(labels, preds),
        "eval_precision": precision_score(labels, preds),
        "eval_recall": recall_score(labels, preds),
        "eval_f1": f1_score(labels, preds),
    }
    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--result_dir", default=os.path.join("results"), type=str,
                        help="To store the result in per epoch")
    ## Other parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a jsonl file).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--model_name_or_path", default="microsoft/unixcoder-base", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_attribution", action='store_true',
                        help="Whether to run attribution")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=123456,
                        help="random seed for initialization")
    parser.add_argument('--validation_metric', type=str, default='f1',
                        help="metric to use to for model selection based on the validation set")
    parser.add_argument('--code_key', type=str, default="input",
                        help="dataset key for code")
    parser.add_argument('--label_key', type=str, default="output",
                        help="dataset key for labels")
    parser.add_argument('--index_key', type=str, default="idx",
                        help="dataset key for index")
    parser.add_argument('--head', type=int,
                        help="number of dataset examples to cut off")
    parser.add_argument("--csv_path", default='results.csv', help="Path to save the CSV results.")
    parser.add_argument('--freeze_encoder_and_embeddings', action='store_true',
                        help="whether to freeze the model's encoder and embeddings layers")
    parser.add_argument('--tokenize_ast_token', type=int, default=0)
    parser.add_argument('--node', type=str, default='node')
    parser.add_argument('--dataset', type=str, default="Devign")
    parser.add_argument('--model_name', type=str, default="causal_vanilla")

    # print arguments
    args = parser.parse_args()
    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args.seed)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    # build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = 2
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    model = Model(model, config, args)
    if args.freeze_encoder_and_embeddings:
        logger.info("Freezing encoder and embeddings parameters")
        model.freeze_encoder_and_embeddings()
    logger.info("Training/evaluation parameters %s", args)

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

        # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        train(args, train_dataset, model, tokenizer)

    # Evaluation
    if args.do_eval:
        output_dir = os.path.join(args.output_dir, 'checkpoint-best-f1/model.bin')
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(torch.load(output_dir))
        result = evaluate(args, model, tokenizer, args.eval_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key] * 100 if "map" in key else result[key], 4)))

    if args.do_test:
        output_dir = os.path.join(args.output_dir, 'checkpoint-best-f1/model.bin')
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(torch.load(output_dir))
        result = test(args, model, tokenizer, args.test_data_file)
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key] * 100 if "map" in key else result[key], 4)))


if __name__ == "__main__":
    main()
