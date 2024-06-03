import sys
import os

sys.path.append('../dataset/')
sys.path.append('../ReGVD/code/')
from attack_utils import CodeDataset, GraphCodeDataset, EPVDDataset
from run import convert_examples_to_features, TextDataset, InputFeatures
from data_augmentation import get_identifiers, get_code_tokens
import json
import logging
import argparse
import warnings
import torch
import time
from attack_utils import build_vocab, set_seed
from model import Model, GNNReGVD
from attack_utils import Recorder
from ptm_attacker import WIR_Attacker
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
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--d_size', type=int, default=128, help="For cnn filter size.")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--cnn_size', type=int, default=128, help="For cnn size.")
    parser.add_argument('--filter_size', type=int, default=3, help="For cnn filter size.")
    parser.add_argument("--gnn", default="ReGCN", type=str, help="ReGCN or ReGGNN")
    parser.add_argument("--feature_dim_size", default=768, type=int,
                        help="feature dim size.")
    parser.add_argument("--hidden_size", default=128, type=int,
                        help="hidden size.")
    parser.add_argument("--num_GNN_layers", default=2, type=int,
                        help="num GNN layers.")
    parser.add_argument("--remove_residual", default=False, action='store_true', help="remove_residual")
    parser.add_argument("--att_op", default='mul', type=str,
                        help="using attention operation for attention: mul, sum, concat")
    parser.add_argument("--num_classes", default=2, type=int,
                        help="num classes.")
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
    config.num_labels = 1 if ptm_model in {"CodeBERT", "ReGVD"} else 2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=False,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

    model = GNNReGVD(model, config, tokenizer, args) if ptm_model == "ReGVD" else Model(model, config, args)
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
    for index, code in enumerate(source_codes):
        code_tokens.append(get_identifiers(code))

    id2token, token2id = build_vocab(code_tokens, 5000)

    recoder = Recorder(args.csv_store_path)
    attacker = WIR_Attacker(args, model, tokenizer, ptm_model, token2id, id2token)

    print("ATTACKER BUILT!")

    adv = {"tokens": [], "raw_tokens": [], "ori_raw": [],
           'ori_tokens': [], "label": [], }
    n_succ = 0.0
    total_cnt = 0
    success_attack = 0
    query_times = 0
    all_start_time = time.time()
    for index, example in enumerate(eval_dataset):
        code = source_codes[index]
        orig_code_tokens = get_code_tokens(code)
        identifiers = get_identifiers(code)
        orig_prob, orig_pred = model.get_results([example], args.eval_batch_size)
        orig_prob = orig_prob[0]
        orig_pred = orig_pred[0]
        label = example[3].item() if ptm_model == "GraphCodeBERT" else example[1].item()
        print_index(index, label, orig_pred)
        print("identifiers: ", identifiers)
        if orig_pred != label:
            recoder.write(index, None, None, None, None, None, None, None, "0")
            continue
        total_cnt += 1
        example_start_time = time.time()
        code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words = attacker.wir_attack(
            example, label, code, label)

        example_end_time = (time.time() - example_start_time) / 60
        print("Example time cost: ", round(example_end_time, 2), "min")
        print("ALL examples time cost: ", round((time.time() - all_start_time) / 60, 2), "min")
        print("Query times in this attack: ", model.query - query_times)
        print("All Query times: ", model.query)
        replace_info = ''
        if replaced_words is not None:
            for key in replaced_words.keys():
                replace_info += key + ':' + replaced_words[key] + ','
        if is_success == 1:
            temp_code_js = {"input": adv_code, "output": label}
            new_feature = convert_examples_to_features(temp_code_js, tokenizer, args)
            if ptm_model == "GraphCodeBERT":
                new_example = GraphCodeDataset([new_feature])
            elif ptm_model == "EPVD":
                new_example = EPVDDataset([new_feature])
            else:
                new_example = CodeDataset([new_feature])
            logits, preds = model.get_results([new_example[0]], args.eval_batch_size)
            adv_pred = preds[0]
            print("attack prediction: ", adv_pred)
            if adv_pred != label:
                print("true adv!")
                success_attack += 1
                recoder.write(index, code, adv_code, len(orig_code_tokens), len(identifiers),
                              replace_info, model.query - query_times, example_end_time, "WIR")
            else:
                print("fake adv!")
                recoder.write(index, None, None, None, None, None, None, None, "0")

        else:
            recoder.write(index, None, None, len(orig_code_tokens), len(identifiers),
                          None, model.query - query_times, example_end_time, "0")

        query_times = model.query
        print("Success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))
    print("Final success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))


if __name__ == "__main__":
    main()