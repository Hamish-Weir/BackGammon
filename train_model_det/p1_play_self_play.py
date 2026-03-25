
import random
import sys
import time
import os

from src.Board import BOARD_SIZE, DIE_SIZE, HOME_SIZE, TOTAL_PLAYER_PIECES, Board
from agents.A0Agent import A0Agent, A0Node

import pickle

start = time.process_time_ns()

PID = sys.argv[1]
SUB_PID = sys.argv[2]

MAX_TURNS = 30
MAX_TIME = 30 * 60 * 10**9

BEST_MODEL_PATH = f"models/{BOARD_SIZE}_{DIE_SIZE}_{HOME_SIZE}_{TOTAL_PLAYER_PIECES}_best_model.pth"
GAME_DATA_PATH =  f"data/self_play_games/{PID}_{SUB_PID}_dataset.pkl"
LEGAL_BOARDS_PATH = f"data/{BOARD_SIZE}_{DIE_SIZE}_{HOME_SIZE}_{TOTAL_PLAYER_PIECES}_boardstates.pkl"

TRAIN_SIMULATIONS = 800
TRAIN_EXPLORATION = 1
TRAIN_C_PUCT = 1
TRAIN_DIRICHLET_ALPHA = 0.1
TRAIN_DIRICHLET_EPSILON = 0.33
TRAIN_TEMPERATURE = 1
TRAIN_TEMPREATURE_PLY = 10



agent1=A0Agent(1,BEST_MODEL_PATH,TRAIN_SIMULATIONS,TRAIN_EXPLORATION,True,TRAIN_DIRICHLET_ALPHA,TRAIN_DIRICHLET_EPSILON,TRAIN_TEMPERATURE,TRAIN_TEMPREATURE_PLY)
agent2=A0Agent(-1,BEST_MODEL_PATH,TRAIN_SIMULATIONS,TRAIN_EXPLORATION,True,TRAIN_DIRICHLET_ALPHA,TRAIN_DIRICHLET_EPSILON,TRAIN_TEMPERATURE,TRAIN_TEMPREATURE_PLY)

players = {
    1: agent1,
    -1: agent2,
}

current_player = 1
board = Board()

def load_boardstates(Boards_Path):
    if not os.path.exists(Boards_Path):
        print(f"{Boards_Path} not found.")
        raise
    else:
        with open(Boards_Path, "rb") as f:
            dataset = pickle.load(f)
        return dataset

legal_boards = load_boardstates(LEGAL_BOARDS_PATH)

if SUB_PID != 1:
    barr,pla = random.choice(legal_boards)
    board.set(barr)
    current_player = pla

turn = 0

opponentMove = None

game_history = []

while True:
    turn += 1

    playerAgent:A0Agent = players[current_player]

    ms = playerAgent.make_move(board, turn, opponentMove)
    
    tensor = A0Node.encode_board(board,current_player)
    prior = playerAgent.get_rollout()
    playee = current_player

    state = (tensor,prior,playee)
    
    game_history.append(state)
    
    board.do_move_sequence(ms,current_player)

    now = time.process_time_ns()
    if board.get_winner() != 0 or turn == MAX_TURNS or (now-start) >= MAX_TIME:
        break
    
    current_player = -current_player

winner = board.get_winner() # win, loss, or turn out

for i, (s, p, pl) in enumerate(game_history):
    if pl == winner:
        value = 1 
    elif -pl == winner:
        value = -1
    else:
        value = -0.5

    game_history[i] = (s, p, value)


with open(GAME_DATA_PATH, "wb") as f:
    pickle.dump(game_history, f)