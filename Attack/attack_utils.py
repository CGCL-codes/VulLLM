from tqdm import tqdm
import csv
import os
import sys
sys.path.append('../dataset')

from data_augmentation import get_example_batch, get_identifiers
import torch
import javalang
import random
from torch.utils.data.dataset import Dataset
import re
import numpy as np


class CodeDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)

class EPVDDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label), torch.tensor(self.examples[i].path_source)


class GraphCodeDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # calculate graph-guided masked function
        attn_mask = np.zeros((512 + 128,
                              512 + 128), dtype=bool)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx])
        max_length = sum([i != 1 for i in self.examples[item].position_idx])
        # sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].input_ids):
            if i in [0, 2]:
                attn_mask[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx):
                    attn_mask[idx + node_index, a + node_index] = True

        return (torch.tensor(self.examples[item].input_ids),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(attn_mask),
                torch.tensor(self.examples[item].label))

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

def get_ann(code):
    ann = {
        "instruction": "Detect whether the following code contains vulnerabilities.",
        "input": code,
    }
    return ann

def get_llm_result(code, model, tokenizer):
    ann = get_ann(code)
    if ann.get("input", "") == "":
        prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
    else:
        prompt = PROMPT_DICT["prompt_input"].format_map(ann)
    model_input = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=model_input['input_ids'],
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=100,
        )
        logits = generation_output.scores
        probabilities = [torch.softmax(logit, dim=-1) for logit in logits]
        prob_dist = probabilities[-2]

        s = generation_output.sequences[0]
        generated_response = tokenizer.decode(s)

        confidence_score, predicted_token_id = torch.max(prob_dist, dim=-1)

        response = generated_response.split("### Response:")[-1].strip()

    return confidence_score.item(), response[0]

def insert_dead_code(code, id2token, code_lines, n_candi):
    adv_codes = []
    identifiers = get_identifiers(code)
    lines = code.split('\n')
    valid_positions = []

    for i, line in enumerate(lines[:-1]):  # 
        next_line = lines[i + 1]
        if is_complete_statement(line) and is_valid_insertion_line(line) and is_valid_insertion_line(next_line):
            if '{' in line or '}' in line:
                match = re.match(r'(\s*)', next_line)
                indent = match.group(1) if match else ''
                valid_positions.append((i + 1, indent))
            else:
                match = re.match(r'(\s*)', line)
                indent = match.group(1) if match else ''
                valid_positions.append((i + 1, indent))

    if not valid_positions:
        return [] 

    total = 0
    times = 0
    while total < n_candi:
        new_lines = lines.copy()
        times += 1
        var = random.sample(id2token, 1)[0]
        line = random.sample(code_lines, 1)[0]
        insert_position, indent = random.choice(valid_positions)
        dead_code = f'{indent}char {var}_2[] = {line};'

        new_lines.insert(insert_position, dead_code)
        if "\n".join(new_lines) not in adv_codes:
            adv_codes.append("\n".join(new_lines))
            total += 1
        if total >= n_candi or times >= 10:
            return adv_codes

def is_complete_statement(line):
    return re.search(r'[;\{\}]$', line.strip())

def is_valid_insertion_line(line):
    return not re.match(r'^\s*[\{\}]?\s*$', line)

def _tokenize(seq, tokenizer):
    seq = seq.replace('\n', '')
    words = seq.split(' ')

    sub_words = []
    keys = []
    index = 0
    for word in words:
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        index += len(sub)

    return words, sub_words, keys

def get_masked_code_by_position(tokens: list, positions: dict):
    '''
    Example:
        tokens: [a,b,c]
        positions: [0,2]
        Return:
            [<mask>, b, c]
            [a, b, <mask>]
    '''
    masked_token_list = []
    replace_token_positions = []
    for variable_name in positions.keys():
        for pos in positions[variable_name]:
            masked_token_list.append(tokens[0:pos] + ['<mask>'] + tokens[pos + 1:])
            replace_token_positions.append(pos)

    return masked_token_list, replace_token_positions

def get_identifier_posistions_from_code(words_list: list, variable_names: list) -> dict:

    positions = {}
    for name in variable_names:
        for index, token in enumerate(words_list):
            if name == token:
                try:
                    positions[name].append(index)
                except:
                    positions[name] = [index]

    return positions

def map_chromesome(chromesome: dict, code: str, lang: str) -> str:
    temp_replace = get_example_batch(code, chromesome)

    return temp_replace


def remove_comments_and_docstrings(source):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    temp=[]
    for x in re.sub(pattern, replacer, source).split('\n'):
        if x.strip()!="":
            temp.append(x)
    return '\n'.join(temp)

def get_code_tokens(code):
    code = remove_comments_and_docstrings(code)
    tokens = javalang.tokenizer.tokenize(code)
    code_tokens = [token.value for token in tokens]
    return code_tokens


def build_vocab(codes, limit=5000):
    vocab_cnt = {"<str>": 0, "<char>": 0, "<int>": 0, "<fp>": 0}
    for c in tqdm(codes):
        for t in c:
            if len(t) > 0:
                if t[0] == '"' and t[-1] == '"':
                    vocab_cnt["<str>"] += 1
                elif t[0] == "'" and t[-1] == "'":
                    vocab_cnt["<char>"] += 1
                elif t[0] in "0123456789.":
                    if 'e' in t.lower():
                        vocab_cnt["<fp>"] += 1
                    elif '.' in t:
                        if t == '.':
                            if t not in vocab_cnt.keys():
                                vocab_cnt[t] = 0
                            vocab_cnt[t] += 1
                        else:
                            vocab_cnt["<fp>"] += 1
                    else:
                        vocab_cnt["<int>"] += 1
                elif t in vocab_cnt.keys():
                    vocab_cnt[t] += 1
                else:
                    vocab_cnt[t] = 1
    vocab_cnt = sorted(vocab_cnt.items(), key=lambda x: x[1], reverse=True)

    idx2txt = ["<unk>"] + ["<pad>"] + [it[0] for index, it in enumerate(vocab_cnt) if index < limit - 1]
    txt2idx = {}
    for idx in range(len(idx2txt)):
        txt2idx[idx2txt[idx]] = idx
    return idx2txt, txt2idx

class Recorder():
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.f = open(file_path, 'w', newline='')
        self.writer = csv.writer(self.f)
        self.writer.writerow(["Index",
                              "Original Code",
                              "Adversarial Code",
                              "Program Length",
                              "Identifier Num",
                              "Replaced Identifiers",
                              "Query Times",
                              "Time Cost",
                              "Type"])

    def write(self, index, code, adv_code, prog_length, nb_var, replace_info, query_times, time_cost, attack_type):
        self.writer.writerow([index,
                              code,
                              adv_code,
                              prog_length,
                              nb_var,
                              replace_info,
                              query_times,
                              time_cost,
                              attack_type])
class Recorder_style():
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.f = open(file_path, 'w')
        self.writer = csv.writer(self.f)
        self.writer.writerow(["Index",
                              "Original Code",
                              "Adversarial Code",
                              "True Label",
                              "Adv Label",
                              "Query Times",
                              "Time Cost"])

    def write(self, index, code, adv_code, true_label, adv_label, query_times, time_cost):
        self.writer.writerow([index,
                              code,
                              adv_code,
                              true_label,
                              adv_label,
                              query_times,
                              time_cost])

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True