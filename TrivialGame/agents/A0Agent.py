from __future__ import annotations
from dataclasses import dataclass, field
import math
import random
from typing import Dict, Optional

import numpy as np

import torch

from networks.A0Network import A0Network
from src.AgentBase import AgentBase
from src.Board import BOARD_END, BOARD_SIZE, BOARD_START, GAME_SIZE, P1BAR, P1OFF, P2BAR, P2OFF, Board

n = BOARD_SIZE+1

@dataclass
class A0Node:
    board: Board
    player: int

    parent: Optional[A0Node]                    = None
    movesequence: Optional[list]                = None
    legal_movesequences: Optional[list[list]]   = None
    
    children:           Dict[tuple, A0Node | None]  = field(default_factory=dict)
    child_prior:        Dict[tuple, float]          = field(default_factory=dict)
    child_visits:       Dict[tuple, int]            = field(default_factory=dict)
    child_total_value:  Dict[tuple, float]          = field(default_factory=dict)
    child_value:        Dict[tuple, float]          = field(default_factory=dict)
    
    def __init__(
        self,
        board: Board,
        player: int,
        parent: Optional[A0Node] = None,
        movesequence: Optional[list] = None
    ):
        self.board = board
        self.player = player
        self.parent = parent
        self.movesequence = movesequence
        
        self.legal_movesequences = self.board.get_legal_movesequences(self.player)

    def best_child(self, exploration: float = 1.4) -> (Optional[A0Node], A0Node): # type: ignore
        best_move_sequence = None
        best_PUCT = -math.inf
        children_visits = sum([self.child_visits.get(ms,0) for ms in self.legal_movesequences])
        for move_sequence in self.legal_movesequences:
            c_key = tuple(move_sequence)
            value = self.child_total_value.get(c_key,0.0)
            prior = self.child_prior.get(c_key,0.0)
            child_visits = self.child_visits.get(c_key,0.0)

            PUCT = (
                value +
                exploration * prior * math.sqrt(children_visits)/(1+child_visits)
            )

            if PUCT > best_PUCT:
                best_PUCT = PUCT
                best_move_sequence = move_sequence

        best_child = self.children.get(best_move_sequence,None)

        return best_child, self
    
    def backup(self, move_sequence, v):
        c_key = tuple(move_sequence)

        total_visits = self.child_visits.get(c_key,0) + 1
        total_value  = self.child_total_value.get(c_key,0.0) + v

        self.child_visits[c_key] = total_visits
        self.child_total_value[c_key] = total_value
        self.value = total_value/total_visits

    @staticmethod
    def movesequence_to_idx(self,movesequence):
        # Note: Function is Bar position dependent 
        L = len(movesequence)
        if L == 2:
            m1,m2 = movesequence[0], movesequence[1]
            s1,d1 = m1
            s2,d2 = m2
            assert s1<=s2
            
            s1id = s1 - BOARD_START + 1 # s1 -> [0..6]
            s2id = s2 - BOARD_START + 1 # s2 -> [0..6]
            eq = (((s1id) * ((n+(n-(s1id-1)))))//2) + (s2id-s1id)

            match (d1,d2):
                case (1,1):
                    return 0 + eq   # 0-27
                case (1,2):
                    return 28 + eq  # 28-55
                case (2,1):
                    return 56 + eq  # 56-83
                case (2,2):
                    return 84 + eq  # 84-111
        elif L == 1:

            m1 = movesequence[0]
            s1,d1 = m1

            s1id = s1 - BOARD_START + 1 # s1 -> [0..6]

            match d1:
                case 1:
                    return 112 + s1id # 112-118
                case 2:
                    return 119 + s1id # 119-125
        else:
            return 126              # 126 (Skip Move)

    @staticmethod
    def flip_movesequence(movesequence, player):
        """
        Converts MS (in perspective p1) 
        to perspective of 'player'
        """
        if player == 1:
            return movesequence
        
        pms = []
        for move in movesequence:
            s,d = move
            ps = (GAME_SIZE - 1) - s
            pms.append((ps,d))

        return pms

    def raw_policy_to_policy_dict(self,policy):
        policy_dict = {}
        for ms in self.legal_movesequences:
            c_key = tuple(ms)
            p_ms = self.flip_movesequence(ms,self.player)
            idx = self.movesequence_to_idx(p_ms)
            policy_dict[c_key] = policy[idx]
        return policy_dict


    def init_val_and_pri(self,model:A0Network):
        state_tensor = self.encode_board()
        val, pol = model[state_tensor]
        
        value = val.item()
        policy = pol.squeeze(0).detach().cpu().numpy()
        
        self.child_prior = self.raw_policy_to_policy_dict(policy) # Initialize child priors

        return value


    def encode_board(self):
        board_arr = self.board._tiles
        
        if self.player == 1:
            player_board = board_arr[::1].copy()
        else:
            player_board = -board_arr[::-1].copy()

        return torch.tensor(player_board, dtype=torch.float32)


    def __str__(self):
        return f"MCTSNode(Player: {self.player:>4d}, Children: {len(self.children.values()):>4d}, Visits: {self.visits:>4d}, Total Value: {self.total_value:>.4f}, Value: {self.value:>.4f}, Action: {self.ms_to_str(self.movesequence,self.parent.player)}\n"
        
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

class A0Agent(AgentBase):
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

    def make_move(self, board: Board, opp_move):
        """Makes a move based on the current board state."""

        root_board = Board()
        root_board.set(board.get())

        self.root = A0Node(board=root_board, player=self.player)
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
    
    def _select(self, node: A0Node) -> A0Node:
        while node.board.get_winner() == 0:
            if node.untried_moves:
                return self._expand(node)
            node = max(node.children.values(), key=lambda c: c.ucb_score(self.c_puct))
        return node

    def _expand(self, node: A0Node) -> A0Node:
        move_sequence_made = node.untried_moves.pop()

        next_board = Board()
        next_board.set(node.board.get())
        next_board.do_move_sequence(move_sequence_made,node.player)

        child = A0Node(
                board           = next_board,
                player          = -node.player,
                parent          = node,
                movesequence    = move_sequence_made,
            )
        
        child.untried_moves = next_board.get_legal_movesequences(child.player)

        node.children[id(child)] = child
        return child

    def _simulate(self, node: A0Node) -> float:
        winner = node.board.get_winner()
        if not winner == 0:
            return 1 if winner == node.player else -1
        return  self._evaluation(node.board, node.player)

    def _backpropagate(self, node: A0Node, v: float):
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