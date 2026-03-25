from datetime import datetime
import os
import pickle
import random
import sys
import time
from typing import Set


import torch

from agents.A0Agent import A0Agent
from agents.RandomAgent import RandomAgent
from agents.MCTSAgent import MCTSAgent
from src.Player import Player
from src.Board import BOARD_SIZE, DIE_SIZE, HOME_SIZE, TOTAL_PLAYER_PIECES, Board

start = time.process_time_ns()

BEST_MODEL_PATH = f"models/{BOARD_SIZE}_{DIE_SIZE}_{HOME_SIZE}_{TOTAL_PLAYER_PIECES}_best_model.pth"
LEGAL_BOARDS_PATH = f"data/{BOARD_SIZE}_{DIE_SIZE}_{HOME_SIZE}_{TOTAL_PLAYER_PIECES}_boardstates.pkl"

A0_SIMS = 800
MCTS_SIMS = 800
MAX_TURNS = 20
MAX_TIME = 20 * 60 * 10**9

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if len(sys.argv) == 4:
    if sys.argv[1] == "A0":
        agent1 = A0Agent(1,BEST_MODEL_PATH,simulations=A0_SIMS)
    elif sys.argv[1] == "MCTS":
        agent1 = MCTSAgent(1,simulations=MCTS_SIMS)
    else:
        agent1 = RandomAgent(1)

    if sys.argv[2] == "A0":
        agent2 = A0Agent(-1,BEST_MODEL_PATH,simulations=A0_SIMS)
    elif sys.argv[2] == "MCTS":
        agent2 = MCTSAgent(-1,simulations=MCTS_SIMS)
    else:
        agent2 = RandomAgent(-1)
    
    print(f"Playing: {sys.argv[1]} vs {sys.argv[2]}")
else:
    agent1 = A0Agent(1,BEST_MODEL_PATH,simulations=A0_SIMS)
    agent2 = A0Agent(-1,BEST_MODEL_PATH,simulations=A0_SIMS)
    print(f"Playing: A0 vs A0")

output_path = sys.argv[3]

player1 = Player(
    "P1",
    agent1,
)

player2 = Player(
    "P2",
    agent2,
)

current_player = 1

players = {
    1: player1,
    -1: player2
}

board = Board()

# def load_boardstates(Boards_Path):
#     if not os.path.exists(Boards_Path):
#         print(f"{Boards_Path} not found.")
#         raise
#     else:
#         with open(Boards_Path, "rb") as f:
#             dataset = pickle.load(f)
#         return dataset

# legal_boards = load_boardstates(LEGAL_BOARDS_PATH)

# barr,pla = random.choice(legal_boards)
# board.set(barr)
# current_player = pla


turn = 0
r = random.randint

opponentMove = None

visited_boards = set()

while True:
    turn += 1

    currentPlayer: Player = players[current_player]
    playerAgent = currentPlayer.agent

    ms = playerAgent.make_move(board, turn, opponentMove)

    board.do_move_sequence(ms,current_player)
    opponentMove = ms

    now = time.process_time_ns()

    if tuple(board.get().flatten()) in visited_boards:
        break
    else:
        visited_boards.add(tuple(board.get().flatten()))

    if board.get_winner() != 0 or turn == MAX_TURNS or (now-start) >= MAX_TIME:
        break

    current_player = -current_player

with open(output_path, "w") as f:
    if sys.argv[1] == "A0":
        f.write(str(board.get_winner()))
        print(board.get_winner())
    elif sys.argv[2] == "A0":
        f.write(str(-board.get_winner()))
        print(-board.get_winner())
