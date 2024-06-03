__licence__ = 'MIT'
__author__ = 'kuyaki'
__credits__ = ['kuyaki']
__maintainer__ = 'kuyaki'
__date__ = '2021/03/23'

from enum import Enum

from tree_sitter import Tree

import os
from pathlib import Path
from typing import Optional

from tree_sitter import Language, Parser, Node

class Lang(Enum):
    JAVA = ".java"
    XML = ".xml"
    PYTHON = ".py"
    C = ".c"

path = 'my-languages.so'
c_path = 'tree-sitter-c'
java_path = 'tree-sitter-java'
Language.build_library(
  path,
  [c_path, java_path]
)
C_LANGUAGE = Language(path, 'c')
JAVA_LANGUAGE = Language(path, 'java')

def tree_sitter_ast(source_code: str, lang: Lang) -> Tree:
    """
    Parse the source code in a specified format into a Tree Sitter AST.
    :param source_code: string with the source code in it.
    :param lang: the source code Lang.
    :return: Tree Sitter AST.
    """
    parser = Parser()

    if lang == Lang.JAVA:
        parser.set_language(JAVA_LANGUAGE)
        return parser.parse(bytes(source_code, "utf8"))
    elif lang == Lang.C:
        parser.set_language(C_LANGUAGE)
        return parser.parse(bytes(source_code, "utf8"))
    else:
        raise NotImplementedError()
