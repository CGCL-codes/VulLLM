import sys
import os

sys.path.append('../dataset/')
sys.path.append('../UniXcoder/code/')
from attack_utils import CodeDataset, GraphCodeDataset, EPVDDataset
from run import convert_examples_to_features, TextDataset, InputFeatures
from data_augmentation import get_identifiers, get_code_tokens
import json
import logging
import argparse
import warnings
import torch
import time
from tree_sitter import Language, Parser
from attack_utils import build_vocab, set_seed, insert_dead_code
from model import Model
from attack_utils import Recorder_style
from ptm_attacker import Style_Attacker
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaModel, RobertaForSequenceClassification, RobertaTokenizer)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)  # Only report warning

logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    'epvd': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}

def print_index(index, label, orig_pred):
    entry = f"Index: {index} | label: {label} | orig_pred: {orig_pred}"
    entry_length = len(entry)
    padding = 100 - entry_length
    left_padding = padding // 2
    right_padding = padding - left_padding
    result = f"{'*' * left_padding} {entry} {'*' * right_padding}"
    print(result)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

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
    parser.add_argument('--d_size', type=int, default=128, help="For cnn filter size.")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--cnn_size', type=int, default=128, help="For cnn size.")
    parser.add_argument('--filter_size', type=int, default=3, help="For cnn filter size.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument('--dropout_probability', type=float, default=0, help='dropout probability')
    args = parser.parse_args()

    args.device = torch.device("cuda")
    # Set seed
    set_seed(args.seed)
    ptm_model = args.output_dir.split("/")[-3]
    args.start_epoch = 0
    args.start_step = 0
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 1 if ptm_model == "CodeBERT" else 2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=False,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

    model = Model(model, config, args)
    checkpoint_prefix = 'checkpoint-best-f1/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    print(output_dir)
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)
    print(f"Loaded tuned model from {output_dir}!")

    ## Load Dataset
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

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
    # attacker = MHM_Attacker(args, model, codebert_mlm, tokenizer_mlm, token2id, id2token)
    attacker = Style_Attacker(args, ptm_model, model, tokenizer)

    print("ATTACKER BUILT!")
    adv = {"tokens": [], "raw_tokens": [], "ori_raw": [],
           'ori_tokens': [], "label": [], }
    n_succ = 0.0
    total_cnt = 0
    success_attack = 0
    start_time = time.time()
    for index, example in enumerate(eval_dataset):
        code = source_codes[index]
        orig_code_tokens = get_code_tokens(code)
        identifiers = get_identifiers(code)
        orig_prob, orig_pred = model.get_results([example], args.eval_batch_size)
        orig_prob = orig_prob[0]
        orig_pred = orig_pred[0]
        label = example[3].item() if ptm_model == "GraphCodeBERT" else example[1].item()
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