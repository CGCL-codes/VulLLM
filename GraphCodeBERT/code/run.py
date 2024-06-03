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
import sys
import pickle
import random
import json
import pandas as pd
import numpy as np
import torch
import functools

from tqdm import tqdm
from tree_sitter import Language, Parser
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.optim import AdamW
from transformers import (get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from model import *

cpu_cont = 16
logger = logging.getLogger(__name__)


def append_to_path(path):
    if path not in sys.path:
        sys.path.append(path)


from parser import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript, DFG_c
from parser import (remove_comments_and_docstrings,
                        tree_to_token_index,
                        index_to_code_token)

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript,
    'c': DFG_c,
}

# load parsers
parsers = {}
lang_path = "parser/my-languages.so"
for lang in dfg_function:
    LANGUAGE = Language(lang_path, lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, parser, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
        # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,
                 label,
                 ):
        # The code function
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg

        # label
        self.label = label


def convert_examples_to_features(js, tokenizer, args):
    parser = parsers["c"]
    code = js["input"]
    if args.tokenize_ast_token:
        code = ' '.join(js['ast_tokens'])
    else:
        code = ' '.join(code.split())
    label = int(js["output"])

    # extract data flow
    code_tokens, dfg = extract_dataflow(code, parser, "c")
    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                   enumerate(code_tokens)]
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
    code_tokens = [y for x in code_tokens for y in x]

    # truncating
    code_tokens = code_tokens[:args.block_size + args.data_flow_length - 3 - min(len(dfg), args.data_flow_length)][
                  :512 - 3]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
    dfg = dfg[:args.block_size + args.data_flow_length - len(source_tokens)]
    source_tokens += [x[0] for x in dfg]
    position_idx += [0 for x in dfg]
    source_ids += [tokenizer.unk_token_id for x in dfg]
    padding_length = args.block_size + args.data_flow_length - len(source_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    source_ids += [tokenizer.pad_token_id] * padding_length

    # reindex
    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
    return InputFeatures(source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg, label)


def get_item(args, examples, item):
    # calculate graph-guided masked function
    attn_mask = np.zeros((args.block_size + args.data_flow_length,
                          args.block_size + args.data_flow_length), dtype=bool)
    # calculate begin index of node and max length of input
    node_index = sum([i > 1 for i in examples[item].position_idx])
    max_length = sum([i != 1 for i in examples[item].position_idx])
    # sequence can attend to sequence
    attn_mask[:node_index, :node_index] = True
    # special tokens attend to all tokens
    for idx, i in enumerate(examples[item].input_ids):
        if i in [0, 2]:
            attn_mask[idx, :max_length] = True
    # nodes attend to code tokens that are identified from
    for idx, (a, b) in enumerate(examples[item].dfg_to_code):
        if a < node_index and b < node_index:
            attn_mask[idx + node_index, a:b] = True
            attn_mask[a:b, idx + node_index] = True
    # nodes attend to adjacent nodes
    for idx, nodes in enumerate(examples[item].dfg_to_dfg):
        for a in nodes:
            if a + node_index < len(examples[item].position_idx):
                attn_mask[idx + node_index, a + node_index] = True

    return (torch.tensor(examples[item].input_ids),
            torch.tensor(examples[item].position_idx),
            torch.tensor(attn_mask),
            torch.tensor(examples[item].label))


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train'):
        self.examples = []
        self.args = args

        with open(file_path) as f:
            datas = json.load(f)
            for data in tqdm(datas):
                self.examples.append(convert_examples_to_features(data, tokenizer, args))

        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                logger.info("position_idx: {}".format(example.position_idx))
                logger.info("dfg_to_code: {}".format(' '.join(map(str, example.dfg_to_code))))
                logger.info("dfg_to_dfg: {}".format(' '.join(map(str, example.dfg_to_dfg))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return get_item(self.args, self.examples, item)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    args.max_steps = args.epochs * len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_perf = 0

    model.zero_grad()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (inputs_ids, position_idx, attn_mask,
             labels) = [x.to(args.device) for x in batch]
            model.train()
            loss, logits = model(input_ids=inputs_ids, position_idx=position_idx, attn_mask=attn_mask, labels=labels)

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss

            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)

        results = evaluate(args, model, tokenizer, eval_when_training=True)

        # Save model checkpoint
        if results[f'eval_{args.validation_metric}'] > best_perf:
            best_perf = results[f'eval_{args.validation_metric}']
            logger.info("  " + "*" * 20)
            logger.info("  Best %s:%s", args.validation_metric, round(best_perf, 4))
            logger.info("  " + "*" * 20)

            checkpoint_prefix = 'checkpoint-best-f1'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer, eval_when_training=False):
    # build dataloaderor(examples[item
    eval_dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        (inputs_ids, position_idx, attn_mask,
         labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(input_ids=inputs_ids, position_idx=position_idx, attn_mask=attn_mask, labels=labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5

    y_preds = logits[:, 1] > best_threshold
    accuracy = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    result = {
        "eval_acc": float(accuracy),
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,

    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, model, tokenizer, best_threshold=0):
    # build dataloader
    eval_dataset = TextDataset(tokenizer, args, file_path=args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        (inputs_ids, position_idx, attn_mask,
         labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(input_ids=inputs_ids, position_idx=position_idx, attn_mask=attn_mask, labels=labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # output result
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = logits[:, 1] > best_threshold
    print(y_trues[:20])
    print(y_preds[:20])
    if not os.path.exists(args.csv_path):
        with open(args.csv_path, 'w') as f:
            f.write('Label,Prediction\n')

    for label, pred in zip(y_trues, y_preds):
        temp_df = pd.DataFrame({'Label': [label], 'Prediction': [pred]})
        temp_df.to_csv(args.csv_path, index=False, mode='a', header=False)
    accuracy = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    result = {
        "eval_acc": float(accuracy),
        "eval_precision": float(precision),
        "eval_recall": float(recall),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,

    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_name_or_path", default="microsoft/graphcodebert-base", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=128, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_attribution", action='store_true',
                        help="Whether to run attribution")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--seed', type=int, default=123456,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=10,
                        help="training epochs")

    parser.add_argument('--validation_metric', type=str, default='f1',
                        help="metric to use to for model selection based on the validation set")
    parser.add_argument('--dataloader_nprocs', type=int, default=6,
                        help="how many processes to use to load data")
    parser.add_argument('--do_preload_datasets', action='store_true',
                        help="whether to go on past loading the datasets")
    parser.add_argument('--parser_lang', type=str, default='c', help="Enter the parser language")
    parser.add_argument('--tokenize_ast_token', type=int, default=0)
    parser.add_argument('--node', type=str, default='node')
    parser.add_argument('--dataset', type=str, default="Devign")
    parser.add_argument("--csv_path", default='results.csv', help="Path to save the CSV results.")
    parser.add_argument('--model_name', type=str, default="causal_vanilla")

    args = parser.parse_args()

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu, )

    # Set seed
    set_seed(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = 2
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    model = Model(model, config, args)
    logger.info("Training/evaluation parameters %s", args)

    if args.do_preload_datasets:
        TextDataset(tokenizer, args, file_path=args.train_data_file)
        TextDataset(tokenizer, args, file_path=args.eval_data_file)
        TextDataset(tokenizer, args, file_path=args.test_data_file)

    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, file_path=args.train_data_file)
        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval:
        output_dir = os.path.join(args.output_dir, 'checkpoint-best-f1/model.bin')
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        evaluate(args, model, tokenizer)

    if args.do_test:
        output_dir = os.path.join(args.output_dir, 'checkpoint-best-f1/model.bin')
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test(args, model, tokenizer, best_threshold=0.5)

    return results


if __name__ == "__main__":
    main()

