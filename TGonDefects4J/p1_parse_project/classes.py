from tabulate import tabulate
from typing import Tuple, Any, List, Generator, Iterator

class Markdown:
    def __init__(self):
        self.__content_list: List[str] = []

    def table(self, data: List[dict]):
        t = tabulate(data, headers="keys", tablefmt="pipe", stralign="center")
        self.__content_list.append(t)
        self.__content_list.append("\n\n")
        return self

    def header(self, content: str, level: int = 1):
        self.__content_list.append(f"\n{'#' * level} {content}\n")
        return self

    def code(self, content: str, lang: str = "java"):
        code_cell = f"```{lang}\n{content}\n```\n"
        self.__content_list.append(code_cell)
        return self

    def text(self, content: str):
        self.__content_list.append(f"{content}\n")
        return self

    def build(self):
        return "".join(self.__content_list)
