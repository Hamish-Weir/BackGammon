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

        self.children:           Dict[tuple, A0Node | None]  = {}
        self.child_prior:        Dict[tuple, float]          = {}
        self.child_visits:       Dict[tuple, int]            = {}
        self.child_total_value:  Dict[tuple, float]          = {}
        self.child_value:        Dict[tuple, float]          = {}

    def best_child(self, exploration: float = 1.4) -> (Optional[A0Node], A0Node): # type: ignore
        best_move_sequence = None
        best_PUCT = -math.inf
        children_visits = sum([self.child_visits.get(tuple(ms),0) for ms in self.legal_movesequences])
        for move_sequence in self.legal_movesequences:
            c_key = tuple(move_sequence)
            value = self.child_total_value.get(c_key,0.0)
            prior = self.child_prior.get(c_key,0.0)
            child_visits = self.child_visits.get(c_key,0.0)

            PUCT = (
                exploration * prior * math.sqrt(children_visits)/(1+child_visits)
                + value
            )

            if PUCT > best_PUCT:
                best_PUCT = PUCT
                best_move_sequence = move_sequence

        c_key = tuple(best_move_sequence)
        best_child = self.children.get(c_key,None)

        return best_child, self, best_move_sequence
    
    def backup(self, move_sequence, v):
        c_key = tuple(move_sequence)

        total_visits = self.child_visits.get(c_key,0) + 1
        total_value  = self.child_total_value.get(c_key,0.0) + v

        self.child_visits[c_key] = total_visits
        self.child_total_value[c_key] = total_value
        self.child_value[c_key] = total_value/total_visits

    @staticmethod
    def _movesequence_to_idx(movesequence):
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
    def _flip_movesequence(movesequence, player):
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

    def _raw_policy_to_policy_dict(self,policy):
        policy_dict = {}
        for ms in self.legal_movesequences:
            c_key = tuple(ms)
            p_ms = self._flip_movesequence(ms,self.player)
            idx = self._movesequence_to_idx(p_ms)
            policy_dict[c_key] = policy[idx]

        total = sum(policy_dict.values())
        policy_dict = {k: v / total for k, v in policy_dict.items()}

        return policy_dict

    def get_val_init_pri(self,model:A0Network):
        state_tensor = self.encode_board()

        model.eval()
        with torch.inference_mode():
            val, pol = model(state_tensor)
        
        value = val.item()
        policy = torch.softmax(pol, dim=-1).squeeze(0).cpu().numpy()
        

        self.child_prior = self._raw_policy_to_policy_dict(policy) # Initialize child priors

        return value

    def encode_board(self):
        board_arr = self.board._tiles
        
        if self.player == 1:
            player_board = board_arr[::1].copy()
        else:
            player_board = -board_arr[::-1].copy()

        return torch.tensor(player_board, dtype=torch.float32)


    def __str__(self):
        if self.parent:
            c_key = tuple(self.movesequence)
            return f"MCTSNode(Player: {self.player:>4d}, Children: {len(self.children.values()):>2d}, Prior: {self.parent.child_prior.get(c_key,0):>.3f}, Visits: {self.parent.child_visits.get(c_key,0):>3d}, Total V: {self.parent.child_total_value.get(c_key,0.0):>.2f}, Value: {self.parent.child_value.get(c_key,0.0):>.5f}, Action: {self.ms_to_str(self.movesequence,self.parent.player)}\n"
        return f"MCTSNode(Player: {self.player:>4d}, Children: {len(self.children.values()):>4d}, Prior: {None}, Visits: {None}, Total Value: {None}, Value: {None}, Action: {None}\n"

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
        model_path = "TrivialGame/models/best_model.pth",
        simulations = 800,
        c_puct = 1,
        training_param = None
    ):
        super().__init__(player)
        

        self.model_path = model_path
        self.model = A0Network()

        self.simulations = simulations
        self.c_puct= c_puct
        self.root = None

        if training_param:
            self.dirichlet_alpha = training_param[0]
            self.dirichlet_epsilon = training_param[1]
            self.temperature_ply = training_param[2]
            self.training = True
        else:
            self.training = False

        try:
            if model_path:
                self.model.load_state_dict(torch.load(model_path,weights_only=True))
        except Exception:
            if self.training:
                print("Model not Found; Initialized with Random Weights")
            else:
                raise Exception(f"Model Not Found ({model_path})")



        

    def make_move(self, board: Board, turn = 0, opp_move = None):
        """Makes a move based on the current board state."""

        root_board = Board()
        root_board.set(board.get())

        self.root = A0Node(board=root_board, player=self.player)
        self.root.get_val_init_pri(self.model)

        for i in range(self.simulations):
            node = self._select(self.root)
            v = self._simulate(node)
            self._backpropagate(node, v)

        if self.root.children:
            best_movesequence = max(self.root.legal_movesequences, key=lambda c: (self.root.child_visits.get(tuple(c),0), self.root.child_value.get(tuple(c),0.0)))
        else: 
            raise Exception("Too Few Simulations Run")
        
        return best_movesequence
    
    def _select(self, node: A0Node) -> A0Node:
        while node.board.get_winner() == 0:
            node, parent, ms = node.best_child(self.c_puct)
            if not node:
                return self._expand(node,parent,ms)
        return node

    def _expand(self, node: A0Node|None, parent:A0Node, movesequence:list) -> A0Node:
        
        next_board = Board()
        next_board.set(parent.board.get())
        next_board.do_move_sequence(movesequence,parent.player)

        child = A0Node(
                board           = next_board,
                player          = -parent.player,
                parent          = parent,
                movesequence    = movesequence,
            )

        c_key = tuple(movesequence)
        parent.children[c_key] = child
        return child

    def _simulate(self, node: A0Node) -> float:
        winner = node.board.get_winner()
        if not winner == 0:
            return 1 if winner == node.player else -1
        return node.get_val_init_pri(self.model) # returns value for newly expanded node

    def _backpropagate(self, node: A0Node, v: float):
        while node.parent:
            v=-v

            node.parent.backup(node.movesequence,v)
            node=node.parent

    def get_rollout(self):
        pol = np.zeros(127)
        root = self.root
        total = 0
        for ms in root.legal_movesequences:
            c_key = tuple(ms)
            n = root.child_visits[c_key]
            total += n

            p_ms  = A0Node._flip_movesequence(ms,root.player)
            p_key = A0Node._movesequence_to_idx(p_ms)
            pol[p_key] = n
        pol = pol/total

        return pol
