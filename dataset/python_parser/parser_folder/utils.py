import collections
import re
from io import StringIO
import  tokenize
def isSameTree(root_p, root_q) -> bool:
    if not root_p and not root_q:
        return True
    if not root_p or not root_q:
        return False

    queue_p = collections.deque([root_p])
    queue_q = collections.deque([root_q])

    while queue_p and queue_q:
        node_p = queue_p.popleft()
        node_q = queue_q.popleft()
        if node_p.type != node_q.type:
            return False
        if len(node_p.children) != len(node_q.children):
            return False
        if len(node_p.children) > 0:
            for child_p, child_q in zip(node_p.children, node_q.children) :
                if child_p.type == child_q.type:
                    queue_p.append(child_p)
                    queue_p.append(child_q)
                else:
                    return False

    return True
def remove_comments_and_docstrings(source,lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
            # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp=[]
        for x in out.split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
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

def tree_to_token_index(root_node):
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        return [(root_node.start_point,root_node.end_point)]
    else:
        code_tokens=[]
        for child in root_node.children:
            code_tokens+=tree_to_token_index(child)
        return code_tokens
    
def tree_to_variable_index(root_node,index_to_code):
    if root_node:
        if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
            index=(root_node.start_point,root_node.end_point)
            _,code=index_to_code[index]
            if root_node.type!=code:
                return [(root_node.start_point,root_node.end_point)]
            else:
                return []
        else:
            code_tokens=[]
            for child in root_node.children:
                code_tokens+=tree_to_variable_index(child,index_to_code)
            return code_tokens  
    else:
        return []

def index_to_code_token(index,code):
    # 开始位置
    start_point=index[0]
    end_point=index[1]
    # 如果在同一行
    if start_point[0]==end_point[0]:
        s=code[start_point[0]][start_point[1]:end_point[1]]
    # 如果多行
    else:
        s=""
        s+=code[start_point[0]][start_point[1]:]
        for i in range(start_point[0]+1,end_point[0]):
            s+=code[i]
        s+=code[end_point[0]][:end_point[1]]   
    return s
