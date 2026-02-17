from __future__ import annotations
from dataclasses import dataclass, field
import math
import random
from typing import Dict, List, Optional

import numpy as np
from src.AgentBase import AgentBase
from src.Board import BOARD_END, BOARD_SIZE, BOARD_START, P1BAR, P1HOME_END, P1HOME_START, P1OFF, P2BAR, P2HOME_END, P2HOME_START, P2OFF, Board

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
            exploration * (math.sqrt(math.log(self.parent.visits) / self.visits))
            - self.value
        )
    
    def backup(self, v):
        self.visits += 1
        self.total_value += v
        self.value = self.total_value/self.visits

    def __str__(self):
        return f"MCTSNode(Player: {self.player:>4d}, Children: {len(self.children.values()):>4d}, Visits: {self.visits:>4d}, Total Value: {self.total_value:>.4f}, Value: {self.value:>.4f}, Action: {self.ms_to_str(self.movesequence,1)}\n"
        

    def __repr__(self) -> str:
        return str(self)
    
    def ms_to_str(self,ms,player):
        if not ms == None:
            ms_str = "["
            for m in ms:
                m_str = self.m_to_str(m,player)
                ms_str += m_str
            ms_str += "]"    
            return ms_str  
        else:
            return "None"

    def m_to_str(self,m,player):
        if m[0] == P1BAR or m[0] == P2BAR:
                s = "BAR"
        else:
            s = f"{m[0]-BOARD_START+1:3d}"

        if Board.end_point(m[0],m[1],player) == P1OFF or Board.end_point(m[0],m[1],player) == P2OFF:
            e = "OFF"
        else:
            e = f"{Board.end_point(m[0],m[1],player)-BOARD_START+1:3d}"

        m_str = str(f"({s}, {e}, {m[1]}), ")
        return m_str

class MCTSAgent(AgentBase):
    def __init__(self, 
        player, 
        simulations = 200,
        c_puct = 1,
        rollouts = 100,
        max_depth = 20,
    ):
        super().__init__(player)
        self.simulations = simulations
        self.c_puct= c_puct
        self.root = None
        self.rollouts = rollouts
        self.max_depth = max_depth
        self.choice = random.choice

    def make_move(self, board: Board, turn = 0, opp_move = None):
        """Makes a move based on the current board state."""

        root_board = Board()
        root_board.set(board.get())

        self.root = MCTSNode(board=root_board, player=self.player)
        self.root.untried_moves = self.root.board.get_legal_movesequences(self.player)

        for i in range(self.simulations):
            node = self._select(self.root)
            v = self._simulate(node)
            self._backpropagate(node, v)

        if self.root.children:
            best_child = max(self.root.children.values(), key=lambda c: (c.visits, c.value))
            best_movesequence = best_child.movesequence
        else: 
            raise Exception("Too Few Simulations Run")
        
        return best_movesequence
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.board.get_winner() == 0:
            if node.untried_moves:
                return self._expand(node)
            node = max(node.children.values(), key=lambda c: c.ucb_score(self.c_puct))
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        move_sequence_made = node.untried_moves.pop()

        next_board = Board()
        next_board.set(node.board.get())
        next_board.do_move_sequence(move_sequence_made,node.player)

        child = MCTSNode(
                board           = next_board,
                player          = -node.player,
                parent          = node,
                movesequence    = move_sequence_made,
            )
        
        child.untried_moves = next_board.get_legal_movesequences(child.player)

        node.children[id(child)] = child
        return child

    def _simulate(self, node: MCTSNode) -> float:
        winner = node.board.get_winner()
        if not winner == 0:
            return 1 if winner == node.player else -1
        return  0 # self._evaluation(node.board, node.player)

    def _backpropagate(self, node: MCTSNode, v: float):
        while node:
            node.backup(v)
            v=-v
            node=node.parent

    def _evaluation(self,board:Board,player):
        return self._rollout(board,player)

    def _rollout(self,board,player):
        def rollout(board,player):
            results = np.empty(self.rollouts, dtype=np.int8)
            arr = board.get()
            b = Board()
            for i in range(self.rollouts):
                b.set(arr)
                depth = 0
                player_turn = player
                turns = 0
                while True:
                    turns+=1
                    legal_ms = b.get_legal_movesequences(player_turn)
                    ms = self.choice(legal_ms)
                    b.do_move_sequence(ms, player_turn)

                    depth += 1
                    if not (b.get_winner() == 0) or depth == self.max_depth:
                        if b.get_winner() == 0:
                            results[i] = 0
                        else:
                            if b.get_winner() == player:
                                results[i] = 1
                            else:
                                results[i] = -1
                        break
                    
                    player_turn = -player_turn
            return results

        results = rollout(board,player)
        win_rate = np.mean(results == 1)
        return win_rate
    