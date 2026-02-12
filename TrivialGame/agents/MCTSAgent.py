from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional

import numpy as np
from src.AgentBase import AgentBase
from src.Board import Board

@dataclass
class MCTSNode:
    board: Board
    player: int

    parent: Optional[MCTSNode] = None
    movesequence: Optional[list] = None

    children: Dict[int, MCTSNode] = field(default_factory=dict)
    visits: int = 0
    total_value: float = 0.0
    value: float = 0.0
    untried_moves: Optional[list[list]] = None

    def ucb_score(self, exploration: float = 1.4) -> float:
        if self.visits == 0:
            return float("inf")
        return (
            self.value / self.visits
            + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        )
    
    def backup(self, value):
        self.visits += 1
        self.total_value += value
        self.value = self.total_value/self.visits

class RandomAgent(AgentBase):
    def __init__(self, 
        player, 
        simulations = 5000,
        c_puct = 1.4
    ):
        super().__init__(player)
        self.simulations = simulations
        self.c_puct= c_puct

    def make_move(self, board: Board, opp_move):
        """Makes a move based on the current board state."""

        root_board = Board()
        root_board.set(board.get())

        self.root = MCTSNode(board=root_board, player=self.player)
        self.root.untried_moves = self.root.board.get_legal_movesequences(self.player)

        for i in range(self.simulations):
            node = self._select(self.root)
            value = self._simulate(node)
            self._backpropagate(node, value)

        if self.root.children:
            best_child = max(self.root.children.values(), key=lambda c: (c.visits, c.value))
            best_movesequence = best_child.movesequence
        else: 
            raise Exception("Too Few Simulations Run")
        
        return best_movesequence
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        pass

    def _expand(self, node: MCTSNode) -> MCTSNode:
        pass

    def _simulate(self, node: MCTSNode) -> float:
        pass

    def _backpropagate(self, node: MCTSNode, value: float):
        pass