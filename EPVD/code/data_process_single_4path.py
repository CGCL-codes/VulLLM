import sys
sys.path.append('/home/EPVD/')
import parserTool.parse as ps
from c_cfg_4path import C_CFG
from parserTool.utils import remove_comments_and_docstrings
from parserTool.parse import Lang
import json
import pickle
import logging
import numpy as np
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}
config_class, model_class, tokenizer_class = MODEL_CLASSES["roberta"]
tokenizer = tokenizer_class.from_pretrained("microsoft/codebert-base", do_lower_case=True)

logger = logging.getLogger(__name__)

def extract_pathtoken(source, path_sequence):
    seqtoken_out = []
    for path in path_sequence:
        seq_code = ''
        for line in path:
            if (line in source):
                seq_code += source[line]
        seqtoken_out.append(seq_code)
        if len(seqtoken_out) > 10:
            break
    if len(path_sequence) == 0:
        seq_code = ''
        for i in source:
            seq_code += source[i]
        seqtoken_out.append(seq_code)
    seqtoken_out = sorted(seqtoken_out, key=lambda i: len(i), reverse=False)
    return seqtoken_out
    
def main():
    output = open('short_4path_cdata.pkl', 'wb')
    path_dict = {}
    state_dict = {}
    num_id = 0
    num_path_dict = {}
    with open("../dataset/cdata/nobalance/train_cdata.jsonl") as f:
        for line in f:
            num_id += 1
            if num_id%100 == 0:
                print(num_id, flush=True)
            
            js = json.loads(line.strip())
            
            clean_code, code_dict = remove_comments_and_docstrings(js['func'], 'c')
            g = C_CFG()
            code_ast = ps.tree_sitter_ast(clean_code, Lang.C)
            s_ast = g.parse_ast_file(code_ast.root_node)
            num_path, cfg_allpath, _, _ = g.get_allpath()
            path_tokens1 = extract_pathtoken(code_dict, cfg_allpath)
            
            path_dict[js['idx']] = path_tokens1, cfg_allpath
    print("train file finish...", flush=True)

    with open("../dataset/cdata/nobalance/valid_cdata.jsonl") as f:
        for line in f:
            num_id += 1
            if num_id%100==0:
                print(num_id, flush=True)
            js = json.loads(line.strip())
            clean_code, code_dict = remove_comments_and_docstrings(js['func'], 'c')
            g = C_CFG()
            code_ast = ps.tree_sitter_ast(clean_code, Lang.C)
            s_ast = g.parse_ast_file(code_ast.root_node)
            num_path, cfg_allpath, _, _ = g.get_allpath()
            path_tokens1 = extract_pathtoken(code_dict, cfg_allpath)
            path_dict[js['idx']] = path_tokens1, cfg_allpath
    print("valid file finish...", flush=True)

    with open("../dataset/cdata/nobalance/test_cdata.jsonl") as f:
        for line in f:
            num_id += 1
            if num_id%100==0:
                print(num_id, flush=True)
            js = json.loads(line.strip())
            clean_code, code_dict = remove_comments_and_docstrings(js['func'], 'c')
            g = C_CFG()
            code_ast = ps.tree_sitter_ast(clean_code, Lang.C)
            s_ast = g.parse_ast_file(code_ast.root_node)
            num_path, cfg_allpath, _, _ = g.get_allpath()
            path_tokens1 = extract_pathtoken(code_dict, cfg_allpath)
            path_dict[js['idx']] = path_tokens1, cfg_allpath
    print("test file finish...", flush=True)

    # Pickle dictionary using protocol 0.
    pickle.dump(path_dict, output)
    output.close()

if __name__=="__main__":
    main()
