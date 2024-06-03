import sys
import os

sys.path.append('../dataset/')
from data_augmentation import get_identifiers, get_code_tokens
import json
import logging
import warnings
import time
from attack_utils import build_vocab, set_seed
from attack_utils import Recorder_style, get_llm_result
from llm_attacker import Style_Attacker, insert_dead_code
import torch
import os
import sys
import json
import copy
import pandas as pd
import argparse
from peft import PeftModel
from transformers import GenerationConfig
from transformers import LlamaTokenizer, CodeLlamaTokenizer, LlamaForCausalLM,AutoTokenizer,AutoModelForCausalLM
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning

logger = logging.getLogger(__name__)

def print_index(index, label, orig_pred):
    entry = f"Index: {index} | label: {label} | orig_pred: {orig_pred}"
    entry_length = len(entry)
    padding = 100 - entry_length
    left_padding = padding // 2
    right_padding = padding - left_padding
    result = f"{'*' * left_padding} {entry} {'*' * right_padding}"
    print(result)

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument('--head', type=int,
                        help="number of dataset examples to cut off")
    parser.add_argument("--base_model", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--csv_store_path", default=None, type=str,
                        help="Base Model")
    parser.add_argument('--code_key', type=str, default="input",
                        help="dataset key for code")
    parser.add_argument('--label_key', type=str, default="output",
                        help="dataset key for labels")
    parser.add_argument('--tokenize_ast_token', type=int, default=0)
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--data_flow_length", default=128, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--tuned_model", required=True, help="Path to the tuned model.")
    parser.add_argument("--eval_data_file", required=True, help="Path to the json file containing test dataset.")
    parser.add_argument("--csv_path", default='results.csv', help="Path to save the CSV results.")

    args = parser.parse_args()

    args.device = torch.device("cuda")
    # Set seed
    set_seed(args.seed)
    args.start_epoch = 0
    args.start_step = 0
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        pad_token_id=tokenizer.eos_token_id
    )
    print("Base model loaded!")

    model = PeftModel.from_pretrained(model, args.tuned_model)
    print("Tuned model loaded!")

    ## Load Dataset
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    with open(args.eval_data_file, "r") as file:
        eval_data = json.load(file)

    source_codes = []
    with open(args.eval_data_file) as f:
        datas = json.load(f)
        for data in datas:
            code = data['input']
            source_codes.append(code)

    code_tokens = []
    code_lines = []
    for index, code in enumerate(source_codes):
        code_tokens.append(get_identifiers(code))
        lines = code.split("\n")
        for line in lines:
            if len(line.split(" ")) > 5:
                code_lines.append(line.strip())

    id2token, token2id = build_vocab(code_tokens, 5000)

    recoder = Recorder_style(args.csv_store_path)
    attacker = Style_Attacker(args, model, tokenizer)

    print("ATTACKER BUILT!")

    adv = {"tokens": [], "raw_tokens": [], "ori_raw": [],
           'ori_tokens': [], "label": [], }
    n_succ = 0.0
    total_cnt = 0
    success_attack = 0

    start_time = time.time()
    for index, example in enumerate(eval_data):
        code = example["input"]
        orig_code_tokens = get_code_tokens(code)
        identifiers = get_identifiers(code)
        orig_prob, orig_pred = get_llm_result(code, model, tokenizer)
        label = example["output"]
        print_index(index, label, orig_pred)

        if orig_pred != label:
            recoder.write(index, None, None, None, None, None, None)
            continue
        total_cnt += 1
        adv_codes = insert_dead_code(code, id2token, code_lines, 5)
        example_start_time = time.time()
        is_success, adv_code, query_times = attacker.style_attack(label, id2token, code_lines, adv_codes)

        if not is_success == 1:
            print("Attack failed on index = {}.".format(index))
            recoder.write(index, None, None, None, None, None, None)
            continue
        example_end_time = (time.time() - example_start_time) / 60
        print("Example time cost: ", round(example_end_time, 2), "min")
        print("ALL examples time cost: ", round((time.time() - start_time) / 60, 2), "min")
        print("Query times in this attack: ", query_times)
        success_attack += 1
        adv_label = 1 if label == 0 else 0
        recoder.write(index, code, adv_code, label, adv_label, query_times, round(example_end_time, 2))
        print("Success rate is : {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))


if __name__ == "__main__":
    main()