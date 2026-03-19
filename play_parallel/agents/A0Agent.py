from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
import math
import random
from typing import Dict, Optional

import numpy as np

import torch

from networks.A0Network import A0Network
from src.AgentBase import AgentBase
from src.Board import BOARD_END, BOARD_SIZE, BOARD_START, DIE_SIZE, GAME_SIZE, HOME_SIZE, P1BAR, P1OFF, P2BAR, P2OFF, PRIOR_ONE_SIZE, PRIOR_ONE_SUB_SIZE, PRIOR_SIZE, PRIOR_TWO_SIZE, PRIOR_TWO_SUB_SIZE, TOTAL_PLAYER_PIECES, Board

@dataclass
class A0Node:
    board: Board
    die1: int
    die2: int
    player: int

    parent: Optional[A0Node]                    = None
    movesequence: Optional[list]                = None
    legal_movesequences: Optional[list[list]]   = None
    
    children:           Dict[tuple, A0Node | None]  = field(default_factory=dict)
    group_prior:        Dict[tuple, float]          = field(default_factory=dict)
    group_visits:       Dict[tuple, int]            = field(default_factory=dict)
    group_total_value:  Dict[tuple, float]          = field(default_factory=dict)
    group_value:        Dict[tuple, float]          = field(default_factory=dict)
    child_visits:       Dict[tuple, int]            = field(default_factory=dict)
    child_total_value:  Dict[tuple, float]          = field(default_factory=dict)
    child_value:        Dict[tuple, float]          = field(default_factory=dict)
    
    def __init__(
        self,
        board: Board,
        die1: int,
        die2: int,
        player: int,
        parent: Optional[A0Node] = None,
        movesequence: Optional[list] = None
    ):
        self.board = board

        self.die1 = die1
        self.die2 = die2
        
        self.player = player
        self.parent = parent
        self.movesequence = movesequence
        
        self.legal_movesequences = self.board.get_legal_movesequences(die1, die2, self.player)

        # MS -> Value
        self.group_prior:       Dict[tuple, float]          = {}
        self.group_visits:      Dict[tuple, int]            = {}
        self.group_total_value: Dict[tuple, int]            = {}
        self.group_value:       Dict[tuple, int]            = {}

        # ((die1,die2),MS) -> Value
        self.children:          Dict[tuple, A0Node | None]  = {}
        self.child_visits:      Dict[tuple, int]            = {}
        self.child_total_value: Dict[tuple, float]          = {}
        self.child_value:       Dict[tuple, float]          = {}
    
    def next_child(self,exploration):  

        # Step 1: find move_sequence with maximum UCB
        best_move_sequence = None
        best_PUCT = -math.inf
        all_visits = sum([self.group_visits.get(tuple(ms),0) for ms in self.legal_movesequences])
        for move_sequence in self.legal_movesequences:
            
            g_key = tuple(move_sequence)
            value = self.group_value.get(g_key,0.0)
            prior = self.group_prior.get(g_key,0.0)
            group_visits = self.group_visits.get(g_key,0.0)

            PUCT = (
                exploration * prior * math.sqrt(all_visits)/(1+group_visits)
                + value
            )

            if PUCT > best_PUCT:
                best_PUCT = PUCT
                best_move_sequence = move_sequence


        # Step 2: find dice pair with minimum total visits
        g_key = tuple(best_move_sequence)
        die1,die2 = min([(i, j) for i in range(1, DIE_SIZE+1) for j in range(1, i+1)],key= lambda k: self.child_visits.get(((k[0],k[1]),g_key),0) * (2 if k[0] == k[1] else 1))

        best_child = self.children.get(((die1,die2),g_key),None)


        return best_child, self, die1,die2, best_move_sequence
    
    def backup(self, move_sequence, die1, die2, v):
        
        c_key = ((die1,die2),tuple(move_sequence))
        g_key = tuple(move_sequence)

        child_total_visits = self.child_visits.get(c_key,0) + 1
        child_total_value  = self.child_total_value.get(c_key,0.0) + v

        self.child_visits[c_key] = child_total_visits
        self.child_total_value[c_key] = child_total_value
        self.child_value[c_key] = child_total_value/child_total_visits

        group_total_visits = self.group_visits.get(g_key,0) + 1
        group_total_value  = self.group_total_value.get(g_key,0.0) + v
        
        self.group_visits[g_key] = group_total_visits
        self.group_total_value[g_key] = group_total_value
        self.group_value[g_key] = group_total_value/child_total_visits

    @staticmethod
    def _movesequence_to_idx(movesequence):
        # Note: Function is Bar position dependent 
            
        def dice_to_idx(die1 = None, die2 = None):
            if die1 and die2:
                d1 = die1-1
                d2 = die2-1
                ds = DIE_SIZE*d1 + d2
                return ds
            if die1:
                return die1-1
            else:
                return 0

        L = len(movesequence)

        if L == 2:
            m1,m2 = movesequence[0], movesequence[1]
            s1,d1 = m1
            s2,d2 = m2

            s1id = s1 - BOARD_START + 1 # s1 -> [0..BOARD_SIZE]
            s2id = s2 - BOARD_START + 1 # s2 -> [0..BOARD_SIZE]
            
            sp1 = (((s1id) * (((BOARD_SIZE+1)+((BOARD_SIZE+1)-(s1id-1)))))//2)
            sp2 = (s2id-s1id)

            subpart =  sp1 + sp2
            part = dice_to_idx(d1,d2)

            return part * PRIOR_TWO_SUB_SIZE + subpart
        
        if L == 1:
            m1 = movesequence[0]
            s1,d1 = m1

            s1id = s1 - BOARD_START+1 # s1 -> [0..BOARD_SIZE]

            return PRIOR_TWO_SIZE + (PRIOR_ONE_SUB_SIZE*dice_to_idx(d1)) + s1id
        
        return PRIOR_TWO_SIZE+PRIOR_ONE_SIZE

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
        legal_logits = []

        # Collect logits for legal moves
        for ms in self.legal_movesequences:
            c_key = tuple(ms)
            p_ms = self._flip_movesequence(ms, self.player)
            idx = self._movesequence_to_idx(p_ms)
            logit = policy[idx]
            policy_dict[c_key] = logit
            legal_logits.append(logit)

        # Convert legal logits to probabilities using softmax
        legal_logits_tensor = torch.tensor(legal_logits, dtype=torch.float32)
        probs = torch.softmax(legal_logits_tensor, dim=0)

        # Assign probabilities back to dictionary
        for i, key in enumerate(policy_dict.keys()):
            policy_dict[key] = probs[i].item()

        return policy_dict

    def get_val_init_pri(self,model:A0Network):
        state_tensor = self.encode_board(self.board,self.die1,self.die2,self.player)

        model.eval()
        with torch.inference_mode():
            val, pol = model(state_tensor)
        
        value = val.item()
        policy = pol.squeeze(0).cpu().numpy()

        self.group_prior = self._raw_policy_to_policy_dict(policy) # Initialize child priors

        return value

    @staticmethod
    def encode_board(board, die1, die2, player):
        board_arr = board._tiles

        counts = np.arange(1, TOTAL_PLAYER_PIECES + 1)[:, None]
        abs_board = np.abs(board_arr)
        piece_mask = (abs_board == counts)

        if player == 1:
            player_board   = ((board_arr > 0) & piece_mask).astype(np.float32)
            opponent_board = ((board_arr < 0) & piece_mask).astype(np.float32)
        else:
            player_board   = ((board_arr < 0) & piece_mask).astype(np.float32)[:, ::-1]
            opponent_board = ((board_arr > 0) & piece_mask).astype(np.float32)[:, ::-1]


        # dice planes
        d1 = max(die1, die2)-1
        d2 = min(die1, die2)-1

        die1_plane = np.full_like(board_arr, d1/(DIE_SIZE-1), dtype=np.float32)
        die2_plane = np.full_like(board_arr, d2/(DIE_SIZE-1), dtype=np.float32)

        player_board   = np.vstack((player_board,die1_plane,die2_plane))
        opponent_board = np.vstack((opponent_board,die1_plane,die2_plane))

        stacked = np.stack((player_board, opponent_board)).astype(np.float32)
        return torch.from_numpy(stacked)

    def __str__(self):
        if self.parent:
            c_key = ((self.die1,self.die2),tuple(self.movesequence))
            return f"MCTSNode(Player: {self.player:>4d}, Children: {len(self.children.values()):>2d}, Prior: {self.parent.group_prior.get(c_key,0):>.3f}, Visits: {self.parent.child_visits.get(c_key,0):>3d}, Total V: {self.parent.child_total_value.get(c_key,0.0):>.2f}, Value: {self.parent.child_value.get(c_key,0.0):>.5f}, Action: {self.die1,self.die2} {self.ms_to_str(self.movesequence,self.parent.player)}\n"
        return f"MCTSNode(Player: {self.player:>4d}, Children: {len(self.children.values()):>4d}, Prior: {None}, Visits: {None}, Total Value: {None}, Value: {None}, Action: {None}\n"

    def __repr__(self) -> str:
        return str(self)
    
    def ms_to_str(self,ms,player):
        if not ms == None:
            ms_str = "["
            # for m in ms:
            #     m_str = self.m_to_str(m,player)
            #     ms_str += m_str
            ms_str += ", ".join([self.m_to_str(m,player) for m in ms])
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

        m_str = str(f"({s}, {e}, {m[1]})")
        return m_str

class A0Agent(AgentBase):

    def __init__(self, 
        player, 
        model_path=f"models_trained/{BOARD_SIZE}_{DIE_SIZE}_{HOME_SIZE}_{TOTAL_PLAYER_PIECES}_A0Model.pth",
        # model_path=f"models/{BOARD_SIZE}_{DIE_SIZE}_{HOME_SIZE}_{TOTAL_PLAYER_PIECES}_best_model.pth",
        simulations = 800,
        c_puct = 1,
        training_on = False,
        dirichlet_alpha = 0.1, # 0.03
        dirichlet_epsilon = 0.25, # 0.25
        temperature = 0,    # 1
        temperature_ply = 0, # 4?
    ):
        super().__init__(player)
        

        self.model_path = model_path
        self.model = A0Network()

        self.simulations = simulations
        self.c_puct= c_puct
        self.root = None

        # training param
        self.training_on = training_on
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature
        self.temperature_ply = temperature_ply

        try:
            if model_path:
                self.model.load_state_dict(torch.load(model_path,weights_only=True))
            else:
                raise Exception("No Model path Provided")
        except Exception:
            self.model._initialize_weights()
            print("Model not Found; Initialized with Random Weights")
            torch.save(self.model.state_dict(), self.model_path)
 
        self.choice = random.choices

    def make_move(self, board: Board, die1: int, die2: int, turn = 0, opp_move = None):
        """Makes a move based on the current board state."""

        temp = board.get_legal_movesequences(die1,die2,self.player)
        if not temp:
            return []

        root_board = Board()
        root_board.set(board.get())


        self.root = A0Node(board=root_board, die1=die1, die2=die2, player=self.player)
        self.root.get_val_init_pri(self.model)

        if self.training_on:
            self._add_noise_to_root()

        for _ in range(self.simulations):
            node = self._select(self.root)
            v = self._simulate(node)
            self._backpropagate(node, v)


        if self.training_on and turn <= self.temperature_ply:
            if self.root.children:
                best_movesequence = self._select_move_with_temperature()
            else: 
                raise Exception("Too Few Simulations Run")
        else:
            if self.root.children:
                best_movesequence = max(self.root.legal_movesequences, key=lambda k: (self.root.group_visits.get(tuple(k),0), self.root.group_value.get(tuple(k),0.0)))
            else: 
                raise Exception("Too Few Simulations Run")
        
        return best_movesequence
    
    def _select(self, node: A0Node) -> A0Node:
        while node.board.get_winner() == 0:
            node, parent, die1, die2, ms = node.next_child(self.c_puct)
            if not node:
                return self._expand(node,parent,die1,die2,ms)
        return node

    def _expand(self, node: A0Node|None, parent:A0Node,die1: int, die2: int, movesequence:list) -> A0Node:
        
        next_board = Board()
        next_board.set(parent.board.get())
        next_board.do_move_sequence(movesequence,parent.player)

        child = A0Node(
                board           = next_board,
                die1            = die1,
                die2            = die2,
                player          = -parent.player,
                parent          = parent,
                movesequence    = movesequence,
            )

        c_key = ((die1,die2),tuple(movesequence))
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

            node.parent.backup(node.movesequence, node.die1, node.die2, v)
            node=node.parent
    
    def _add_noise_to_root(self):

        root = self.root

        L = len(root.legal_movesequences)

        orig_probs = np.array([root.group_prior.get(tuple(m), 0.0) for m in root.legal_movesequences])
        # normalize just in case
        s = orig_probs.sum()
        if s <= 0:
            # fallback uniform
            orig_probs = np.ones_like(orig_probs) / L
        else:
            orig_probs = orig_probs / s

        # sample dirichlet noise
        if self.dirichlet_alpha > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * L)
            mixed = (1 - self.dirichlet_epsilon) * orig_probs + self.dirichlet_epsilon * noise
        else:
            # dirichlet_alpha = 0 -> no noise
            mixed = orig_probs

        # write back into node.P for the legal moves
        for m, p in zip(root.legal_movesequences, mixed):
            root.group_prior[tuple(m)] = np.float32(p)

        # ensure normalization
        s2 = sum(root.group_prior.values())
        for k in root.group_prior:
            root.group_prior[k] = root.group_prior[k] / s2

    def _select_move_with_temperature(self):
        moves = self.root.legal_movesequences
        visits = [self.root.group_visits.get(tuple(c), 0) for c in moves]

        # If nothing has been visited yet, just pick randomly
        if sum(visits) == 0:
            return random.choice(moves)

        # If temperature is 0, fall back to greedy
        if self.temperature <= 0:
            return moves[visits.index(max(visits))]

        # Temperature-scaled sampling
        scaled = [v ** (1.0 / self.temperature) for v in visits]
        total = sum(scaled)
        probs = [v / total for v in scaled]

        return random.choices(moves, weights=probs, k=1)[0]
    
    def get_rollout(self):
        pol = np.zeros(PRIOR_SIZE)
        root = self.root
        total = 0
        for ms in root.legal_movesequences:
            g_key = tuple(ms)
            n = root.group_visits.get(g_key,0)
            total += n
            p_ms  = A0Node._flip_movesequence(ms,root.player)
            p_key = A0Node._movesequence_to_idx(p_ms)
            pol[p_key] = n
        pol = pol/total

        return pol