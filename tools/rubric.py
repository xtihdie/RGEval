from __future__ import annotations
from typing import List


class RubricNode:
    def __init__(
            self,
            node_id: str = "",
            name: str = "",
            is_leaf: bool = False,
            criteria: str = "",
            children: List[RubricNode] = None,
            score: float = 0,
    ):
        self.id = node_id
        self.name = name
        self.is_leaf = is_leaf
        self.criteria = criteria
        self.children = children
        self.score = score

        if self.children is None:
            self.children = []

    def set_id(self, node_id: str):
        self.id = node_id

    def set_name(self, name: str):
        self.name = name

    def set_is_leaf(self, is_leaf: bool):
        self.is_leaf = is_leaf

    def set_criteria(self, criteria: str):
        self.criteria = criteria

    def get_criteria(self):
        return self.criteria

    def append_children(self, children: RubricNode):
        self.children.append(children)

    def extend_childrren(self, children: List[RubricNode]):
        self.children.extend(children)

    def set_score(self, score: float):
        self.score = score
