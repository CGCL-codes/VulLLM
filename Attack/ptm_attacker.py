import sys
import os

sys.path.append('../dataset/')
sys.path.append('../ReGVD/code/')
from run import convert_examples_to_features, TextDataset, InputFeatures
from data_augmentation import get_identifiers, get_example,get_code_tokens
import csv
import subprocess
import copy
import re
import json
import logging
import argparse
import warnings
from data_utils import is_valid_identifier
from attack_utils import get_masked_code_by_position, _tokenize, insert_dead_code
from attack_utils import map_chromesome, EPVDDataset, CodeDataset, GraphCodeDataset, get_identifier_posistions_from_code
import torch
import numpy as np
import random
from model import Model
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset

def get_importance_score(args, ptm_model, example, code, label, words_list: list, sub_words: list, variable_names: list, tgt_model,
                         tokenizer, label_list, batch_size=16, max_length=512, model_type='classification'):
    '''Compute the importance score of each variable'''
    # label: example[1] tensor(1)
    positions = get_identifier_posistions_from_code(words_list, variable_names)
    if len(positions) == 0:
        return None, None, None

    new_example = []

    masked_token_list, replace_token_positions = get_masked_code_by_position(words_list, positions)

    for index, tokens in enumerate([words_list] + masked_token_list):
        new_code = ' '.join(tokens)
        temp_code_js = {"input": new_code, "output": label}
        new_feature = convert_examples_to_features(temp_code_js, tokenizer, args)
        new_example.append(new_feature)
    new_dataset = None
    if ptm_model == "GraphCodeBERT":
        new_dataset = GraphCodeDataset(new_example)
    elif ptm_model == "EPVD":
        new_dataset = EPVDDataset(new_example)
    else:
        new_dataset = CodeDataset(new_example)
    logits, preds = tgt_model.get_results(new_dataset, args.eval_batch_size)
    orig_probs = logits[0]
    orig_label = preds[0]

    orig_prob = max(orig_probs)

    importance_score = []
    for prob in logits[1:]:
        importance_score.append(orig_prob - prob[orig_label])

    return importance_score, replace_token_positions, positions

class MHM_Attacker():
    def __init__(self, args, model_tgt, _ptm_model, _token2idx, _idx2token) -> None:
        self.ptm_model = _ptm_model
        self.classifier = model_tgt
        self.token2idx = _token2idx
        self.idx2token = _idx2token
        self.args = args

    def mcmc_random(self, tokenizer, code=None, _label=None, _n_candi=30,
                    _max_iter=100, _prob_threshold=0.95):
        identifiers = get_identifiers(code)
        code_tokens = get_code_tokens(code)
        processed_code = " ".join(code_tokens)
        prog_length = len(code_tokens)
        words, sub_words, keys = _tokenize(processed_code, tokenizer)
        raw_tokens = copy.deepcopy(words)

        uid = get_identifier_posistions_from_code(words, identifiers)

        if len(uid) <= 0:  
            return {'succ': None, 'tokens': None, 'raw_tokens': None}

        variable_substitue_dict = {}
        for tgt_word in uid.keys():
            variable_substitue_dict[tgt_word] = random.sample(self.idx2token, _n_candi)

        old_uids = {}
        old_uid = ""
        for iteration in range(1, 1 + _max_iter):
            res = self.__replaceUID_random(tokenizer, _tokens=code, _label=_label, _uid=uid,
                                           substitute_dict=variable_substitue_dict,
                                           _n_candi=_n_candi,
                                           _prob_threshold=_prob_threshold)
            self.__printRes(_iter=iteration, _res=res, _prefix="  >> ")

            if res['status'].lower() in ['s', 'a']:
                if iteration == 1:
                    old_uids[res["old_uid"]] = []
                    old_uids[res["old_uid"]].append(res["new_uid"])
                    old_uid = res["old_uid"]

                flag = 0
                for k in old_uids.keys():
                    if res["old_uid"] == old_uids[k][-1]:
                        flag = 1
                        old_uids[k].append(res["new_uid"])
                        old_uid = k
                        break
                if flag == 0:
                    old_uids[res["old_uid"]] = []
                    old_uids[res["old_uid"]].append(res["new_uid"])
                    old_uid = res["old_uid"]

                code = res['tokens']
                uid[res['new_uid']] = uid.pop(res['old_uid'])  
                variable_substitue_dict[res['new_uid']] = variable_substitue_dict.pop(res['old_uid'])

                for i in range(len(raw_tokens)):
                    if raw_tokens[i] == res['old_uid']:
                        raw_tokens[i] = res['new_uid']
                if res['status'].lower() == 's':
                    replace_info = {}
                    nb_changed_pos = 0
                    for uid_ in old_uids.keys():
                        replace_info[uid_] = old_uids[uid_][-1]
                        nb_changed_pos += len(uid[old_uids[uid_][-1]])
                    return {'succ': True, 'tokens': code,
                            'raw_tokens': raw_tokens, "prog_length": prog_length, "new_pred": res["new_pred"],
                            "is_success": 1, "old_uid": old_uid, "score_info": res["old_prob"][0] - res["new_prob"][0],
                            "nb_changed_var": len(old_uids), "nb_changed_pos": nb_changed_pos,
                            "replace_info": replace_info, "attack_type": "MHM-Origin"}
        replace_info = {}
        nb_changed_pos = 0

        for uid_ in old_uids.keys():
            replace_info[uid_] = old_uids[uid_][-1]
            nb_changed_pos += len(uid[old_uids[uid_][-1]])

        return {'succ': False, 'tokens': res['tokens'], 'raw_tokens': None, "prog_length": prog_length,
                "new_pred": res["new_pred"], "is_success": -1, "old_uid": old_uid,
                "score_info": res["old_prob"][0] - res["new_prob"][0], "nb_changed_var": len(old_uids),
                "nb_changed_pos": nb_changed_pos, "replace_info": replace_info, "attack_type": "MHM-Origin"}

    def __replaceUID_random(self, tokenizer, _tokens, _label=None, _uid={}, substitute_dict={},
                            _n_candi=30, _prob_threshold=0.95, _candi_mode="random"):

        assert _candi_mode.lower() in ["random", "nearby"]
        selected_uid = random.sample(list(substitute_dict.keys()), 1)[0]  
        if _candi_mode == "random":
            # First, generate candidate set.
            # The transition probabilities of all candidate are the same.
            candi_token = [selected_uid]
            candi_tokens = [copy.deepcopy(_tokens)]
            candi_labels = [_label]
            for c in random.sample(self.idx2token, _n_candi):  
                if c in _uid.keys():
                    continue
                if is_valid_identifier(c):  
                    candi_token.append(c)
                    candi_tokens.append(copy.deepcopy(_tokens))
                    candi_labels.append(_label)
                    candi_tokens[-1] = get_example(candi_tokens[-1], selected_uid, c)

            new_example = []
            for tmp_tokens in candi_tokens:
                tmp_code = tmp_tokens
                temp_code_js = {"input": tmp_code, "output": _label}
                new_feature = convert_examples_to_features(temp_code_js, tokenizer, self.args)
                new_example.append(new_feature)
            new_dataset = None
            if self.ptm_model == "GraphCodeBERT":
                new_dataset = GraphCodeDataset(new_example)
            elif self.ptm_model == "EPVD":
                new_dataset = EPVDDataset(new_example)
            else:
                new_dataset = CodeDataset(new_example)
            prob, pred = self.classifier.get_results(new_dataset, self.args.eval_batch_size)

            for i in range(len(candi_token)):  # Find a valid example
                if pred[i] != _label:  
                    return {"status": "s", "alpha": 1, "tokens": candi_tokens[i],
                            "old_uid": selected_uid, "new_uid": candi_token[i],
                            "old_prob": prob[0], "new_prob": prob[i],
                            "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}

            candi_idx = 0
            min_prob = 1.0

            for idx, a_prob in enumerate(prob[1:]):
                if a_prob[_label] < min_prob:
                    candi_idx = idx + 1
                    min_prob = a_prob[_label]

            # At last, compute acceptance rate.
            alpha = (1 - prob[candi_idx][_label] + 1e-10) / (1 - prob[0][_label] + 1e-10)
            if random.uniform(0, 1) > alpha or alpha < _prob_threshold:
                return {"status": "r", "alpha": alpha, "tokens": candi_tokens[i],
                        "old_uid": selected_uid, "new_uid": candi_token[i],
                        "old_prob": prob[0], "new_prob": prob[i],
                        "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}
            else:
                return {"status": "a", "alpha": alpha, "tokens": candi_tokens[i],
                        "old_uid": selected_uid, "new_uid": candi_token[i],
                        "old_prob": prob[0], "new_prob": prob[i],
                        "old_pred": pred[0], "new_pred": pred[i], "nb_changed_pos": _tokens.count(selected_uid)}
        else:
            pass

    def __printRes(self, _iter=None, _res=None, _prefix="  => "):
        if _res['status'].lower() == 's':  # Accepted & successful
            print("%s iter %d, SUCC! %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                  (_prefix, _iter, _res['old_uid'], _res['new_uid'],
                   _res['old_pred'], _res['new_pred'],
                   _res['old_prob'][_res['old_pred']],
                   _res['new_prob'][_res['old_pred']], _res['alpha']), flush=True)
        elif _res['status'].lower() == 'r':  # Rejected
            print("%s iter %d, REJ. %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                  (_prefix, _iter, _res['old_uid'], _res['new_uid'],
                   _res['old_pred'], _res['new_pred'],
                   _res['old_prob'][_res['old_pred']],
                   _res['new_prob'][_res['old_pred']], _res['alpha']), flush=True)
        elif _res['status'].lower() == 'a':  # Accepted
            print("%s iter %d, ACC! %s => %s (%d => %d, %.5f => %.5f) a=%.3f" % \
                  (_prefix, _iter, _res['old_uid'], _res['new_uid'],
                   _res['old_pred'], _res['new_pred'],
                   _res['old_prob'][_res['old_pred']],
                   _res['new_prob'][_res['old_pred']], _res['alpha']), flush=True)


class WIR_Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, ptm_model, _token2idx, _idx2token) -> None:
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.ptm_model = ptm_model
        self.token2idx = _token2idx
        self.idx2token = _idx2token
        self.args = args

    def wir_attack(self, example, true_label, code, label):
        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        adv_code = ''
        temp_label = None

        identifiers = get_identifiers(code)
        code_tokens = get_code_tokens(code)
        prog_length = len(code_tokens)

        processed_code = " ".join(code_tokens)

        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_tgt)
        variable_names = identifiers
        if not orig_label == true_label:
            is_success = -4
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        if len(variable_names) == 0:
            is_success = -3
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        sub_words = [self.tokenizer_tgt.cls_token] + sub_words[:self.args.block_size - 2] + [
            self.tokenizer_tgt.sep_token]

        importance_score, replace_token_positions, names_positions_dict = get_importance_score(self.args, self.ptm_model, example,
                                                                                               processed_code, label,
                                                                                               words,
                                                                                               sub_words,
                                                                                               variable_names,
                                                                                               self.model_tgt,
                                                                                               self.tokenizer_tgt,
                                                                                               [0, 1],
                                                                                               batch_size=self.args.eval_batch_size,
                                                                                               max_length=self.args.block_size,
                                                                                               model_type='classification')

        if importance_score is None:
            return code, prog_length, adv_code, true_label, orig_label, temp_label, -3, variable_names, None, None, None, None

        token_pos_to_score_pos = {}

        for i, token_pos in enumerate(replace_token_positions):
            token_pos_to_score_pos[token_pos] = i
        names_to_importance_score = {}

        for name in names_positions_dict.keys():
            total_score = 0.0
            positions = names_positions_dict[name]
            for token_pos in positions:
                total_score += importance_score[token_pos_to_score_pos[token_pos]]

            names_to_importance_score[name] = total_score

        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=True)

        final_words = copy.deepcopy(words)
        final_code = copy.deepcopy(code)
        nb_changed_var = 0  
        nb_changed_pos = 0
        is_success = -1
        replaced_words = {}

        for name_and_score in sorted_list_of_names[:20]:
            tgt_word = name_and_score[0]
            tgt_positions = names_positions_dict[tgt_word]

            all_substitues = []
            num = 0
            while num < 5:
                tmp_var = random.choice(self.idx2token)
                if is_valid_identifier(tmp_var):
                    all_substitues.append(tmp_var)
                    num += 1


            most_gap = 0.0
            candidate = None
            replace_examples = []

            substitute_list = []
            for substitute in all_substitues:
                # temp_replace = copy.deepcopy(final_words)
                # for one_pos in tgt_positions:
                #     temp_replace[one_pos] = substitute

                substitute_list.append(substitute)

                temp_code = get_example(final_code, tgt_word, substitute)
                temp_code_js = {"input": temp_code, "output": true_label}

                new_feature = convert_examples_to_features(temp_code_js, self.tokenizer_tgt, self.args)
                replace_examples.append(new_feature)
            if len(replace_examples) == 0:
                continue
            new_dataset = None
            if self.ptm_model == "GraphCodeBERT":
                new_dataset = GraphCodeDataset(replace_examples)
            elif self.ptm_model == "EPVD":
                new_dataset = EPVDDataset(replace_examples)
            else:
                new_dataset = CodeDataset(replace_examples)
            logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
            assert (len(logits) == len(substitute_list))

            for index, temp_prob in enumerate(logits):
                temp_label = preds[index]
                if temp_label != orig_label:
                    is_success = 1
                    nb_changed_var += 1
                    nb_changed_pos += len(names_positions_dict[tgt_word])
                    candidate = substitute_list[index]
                    replaced_words[tgt_word] = candidate
                    adv_code = get_example(final_code, tgt_word, candidate)
                    print("%s SUC! %s => %s (%.5f => %.5f)" % \
                          ('>>', tgt_word, candidate,
                           current_prob,
                           temp_prob[orig_label]), flush=True)
                    return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
                else:
                    gap = current_prob - temp_prob[temp_label]
                    if gap > most_gap:
                        most_gap = gap
                        candidate = substitute_list[index]

            if most_gap > 0:

                nb_changed_var += 1
                nb_changed_pos += len(names_positions_dict[tgt_word])
                current_prob = current_prob - most_gap
                replaced_words[tgt_word] = candidate
                final_code = get_example(final_code, tgt_word, candidate)
                print("%s ACC! %s => %s (%.5f => %.5f)" % \
                      ('>>', tgt_word, candidate,
                       current_prob + most_gap,
                       current_prob), flush=True)
            else:
                replaced_words[tgt_word] = tgt_word

            adv_code = final_code

        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words

class Style_Attacker():
    def __init__(self, args, ptm_model, model_tgt, tokenizer_tgt) -> None:
        self.args = args
        self.ptm_model = ptm_model
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt

    def style_attack(self, true_label, id2token, code_lines, adv_codes):
        # print("adv_codes", adv_codes)
        query_times = 0
        is_success = -1
        total_adv_codes = []
        new_adv_codes = []
        total_adv_codes.extend(adv_codes)
        for tmp_code in adv_codes:
            new_adv_codes = insert_dead_code(tmp_code, id2token, code_lines, 5)
            total_adv_codes.extend(new_adv_codes)
            for tmp_code2 in new_adv_codes:
                new_adv_codes2 = insert_dead_code(tmp_code2, id2token, code_lines, 3)
                total_adv_codes.extend(new_adv_codes2)

        # print("adv_codes1", adv_codes1)
        for tmp_code in total_adv_codes:
            temp_code_js = {"input": tmp_code, "output": true_label}
            new_feature = convert_examples_to_features(temp_code_js, self.tokenizer_tgt, self.args)
            new_dataset = None
            if self.ptm_model == "GraphCodeBERT":
                new_dataset = GraphCodeDataset([new_feature])
            elif self.ptm_model == "EPVD":
                new_dataset = EPVDDataset([new_feature])
            else:
                new_dataset = CodeDataset([new_feature])
            logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
            temp_label = preds[0]
            query_times += 1
            if temp_label != true_label:
                is_success = 1
                print("%s SUC! (%.5f => %.5f)" % \
                      ('>>', true_label, temp_label), flush=True)
                return is_success, tmp_code, query_times

        return is_success, None, None