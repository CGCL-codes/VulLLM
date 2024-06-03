# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

from tree_sitter import Language, Parser
from .utils import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)

def DFG_c(root_node, index_to_code, states):
    assignment = ['assignment_expression']
    def_statement = ['init_declatator', 'pointer_declarator', 'array_declarator']
    increment_statement = ['update_expression']
    if_statement = ['if_statement', 'else']
    for_statement = ['for_statement']
    while_statement = ['while_statement']
    parameter_statement = ['parameter_declaration']
    do_first_statement = []
    states = states.copy()
    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        idx, code = index_to_code[(root_node.start_point, root_node.end_point)]
        if root_node.type == code or (root_node.parent.type == 'function_declarator' and root_node):
            return [], states
        elif code in states:
            return [(code, idx, 'comesFrom', [code], states[code].copy())], states
        elif root_node.type == 'identifier':
            if root_node.parent.type == 'declaration':
                states[code]=[idx]
                return [(code,idx,'comesFrom',[],[])],states
            return [], states
        else:
            return [], states
    elif root_node.type in def_statement:

        if root_node.parent.type == 'function_definition':
            while root_node.type == 'pointer_declarator' and root_node.child_by_field_name('declarator').type == 'pointer_declarator':
                root_node = root_node.child_by_field_name('declarator')
            DFG = []
            for child in root_node.children:
                if child.type not in do_first_statement:
                    temp, states = DFG_c(child, index_to_code, states)
                    DFG += temp
            return sorted(DFG, key=lambda x: x[1]), states
        name = root_node.child_by_field_name('declarator')
        value = root_node.child_by_field_name('value')
        DFG = []
        if value is None:
            indexs = tree_to_variable_index(name, index_to_code)
            for index in indexs:
                idx, code = index_to_code[index]
                DFG.append((code, idx, 'comesFrom', [], []))
                states[code] = [idx]
            return sorted(DFG, key=lambda x: x[1]), states
        else:
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            temp, states = DFG_c(value, index_to_code, states)
            DFG += temp
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    DFG.append((code1, idx1, 'comesFrom', [code2], [idx2]))
                states[code1] = [idx1]
            return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in assignment:
        # left_nodes = root_node.child_by_field_name('left')
        # right_nodes = root_node.child_by_field_name('right')
        # DFG = []
        # temp, states = DFG_c(right_nodes, index_to_code, states)
        # DFG += temp
        # # filter field identifiers
        # while left_nodes.type == 'field_expression' or left_nodes.type == 'subscript_expression':
        #     left_nodes = left_nodes.child_by_field_name('argument')
        # left_node = left_nodes
        # name_indexs = tree_to_variable_index(left_node, index_to_code)
        # value_indexs = tree_to_variable_index(right_nodes, index_to_code)
        # for index1 in name_indexs:
        #     idx1, code1 = index_to_code[index1]
        #     for index2 in value_indexs:
        #         idx2, code2 = index_to_code[index2]
        #         if code1 == "alarm_timers":
        #             print(12)
        #         if code1 in
        #         DFG.append((code1, idx1, 'computedFrom', [code2], [idx2]))
        #     states[code1] = [idx1]
        return [], states
    elif root_node.type in increment_statement:
        DFG = []
        indexs = tree_to_variable_index(root_node, index_to_code)
        for index1 in indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in indexs:
                idx2, code2 = index_to_code[index2]
                DFG.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            states[code1] = [idx1]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in if_statement:
        DFG = []
        current_states = states.copy()
        others_states = []
        flag = False
        tag = False
        if 'else' in root_node.type:
            tag = True
        for child in root_node.children:
            if 'else' in child.type:
                tag = True
            if child.type not in if_statement and flag is False:
                temp, current_states = DFG_c(child, index_to_code, current_states)
                DFG += temp
            else:
                flag = True
                temp, new_states = DFG_c(child, index_to_code, states)
                DFG += temp
                others_states.append(new_states)
        others_states.append(current_states)
        if tag is False:
            others_states.append(states)
        new_states = {}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key] = dic[key].copy()
                else:
                    new_states[key] += dic[key]
        for key in states:
            if key not in new_states:
                new_states[key] = states[key]
            else:
                new_states[key] += states[key]
        for key in new_states:
            new_states[key] = sorted(list(set(new_states[key])))
        return sorted(DFG, key=lambda x: x[1]), new_states
    elif root_node.type in for_statement:
        DFG = []
        for child in root_node.children:
            temp, states = DFG_c(child, index_to_code, states)
            DFG += temp
        flag = False
        for child in root_node.children:
            if flag:
                temp, states = DFG_c(child, index_to_code, states)
                DFG += temp
            elif child.type == "variable_declaration":
                flag = True
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(set(dic[(x[0], x[1], x[2])][0] + x[3]))
                dic[(x[0], x[1], x[2])][1] = sorted(list(set(dic[(x[0], x[1], x[2])][1] + x[4])))
        DFG = [(x[0], x[1], x[2], y[0], y[1]) for x, y in sorted(dic.items(), key=lambda t: t[0][1])]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in while_statement:
        DFG = []
        for i in range(2):
            for child in root_node.children:
                temp, states = DFG_c(child, index_to_code, states)
                DFG += temp
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(set(dic[(x[0], x[1], x[2])][0] + x[3]))
                dic[(x[0], x[1], x[2])][1] = sorted(list(set(dic[(x[0], x[1], x[2])][1] + x[4])))
        DFG = [(x[0], x[1], x[2], y[0], y[1]) for x, y in sorted(dic.items(), key=lambda t: t[0][1])]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in parameter_statement:
        child = root_node.child_by_field_name('declarator')
        if not child:
            return [], states
        while(child.type != 'identifier'):
            if child.type == 'parenthesized_declarator':
                child = child.children[1]
            else:
                child = child.child_by_field_name('declarator')
            if not child:
                return [], states
        idx,code=index_to_code[(child.start_point,child.end_point)]
        states[code]=[idx]
        return [(code,idx,'comesFrom',[],[])],states
    else:
        DFG = []
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp, states = DFG_c(child, index_to_code, states)
                DFG += temp
        return sorted(DFG, key=lambda x: x[1]), states
