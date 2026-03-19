from datetime import datetime
import os
import random
import sys
import time

import torch

from agents.A0Agent import A0Agent
from agents.RandomAgent import RandomAgent
from agents.MCTSAgent import MCTSAgent
from src.Player import Player
from src.Board import DIE_SIZE, Board

start = time.process_time_ns()

A0_SIMS = 200
MCTS_SIMS = 200
MAX_TURNS = 60
MAX_TIME = 15 * 60 * 10**9

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if len(sys.argv) == 3:
    if sys.argv[1] == "A0":
        agent1 = A0Agent(1,simulations=A0_SIMS)
    elif sys.argv[1] == "MCTS":
        agent1 = MCTSAgent(1,simulations=MCTS_SIMS)
    else:
        agent1 = RandomAgent(1)

    if sys.argv[2] == "A0":
        agent2 = A0Agent(-1,simulations=A0_SIMS)
    elif sys.argv[2] == "MCTS":
        agent2 = MCTSAgent(-1,simulations=MCTS_SIMS)
    else:
        agent2 = RandomAgent(-1)
else:
    agent1 = A0Agent(1,simulations=A0_SIMS)
    agent2 = A0Agent(-1,simulations=A0_SIMS)

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
turn = 0
r = random.randint

opponentMove = None

while True:
    turn += 1

    currentPlayer: Player = players[current_player]
    playerAgent = currentPlayer.agent

    die1 = r(1,DIE_SIZE)
    die2 = r(1,DIE_SIZE)

    ms = playerAgent.make_move(board, die1, die2, turn, opponentMove)

    board.do_move_sequence(ms,current_player)
    opponentMove = ms

    now = time.process_time_ns()
    if board.get_winner() != 0 or turn == MAX_TURNS or (now-start) >= MAX_TIME:
        break

    current_player = -current_player

print(board.get_winner())
