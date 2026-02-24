
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import os
import time
import traceback
import multiprocessing as mp

import torch

from agents.A0Agent import A0Agent
from agents.RandomAgent import RandomAgent
from agents.MCTSAgent import MCTSAgent
from src.Player import Player
from src.Board import Board

MAX_TURNS = 20

def play_game(p):
    global MAX_TURNS
    print(f"Process {p:>8d} Started")
    try:
        player2 = Player(
            "P1",
            A0Agent(-1,"models/best_model.pth",10),
            
        )

        player1 = Player(
            "P2",
            MCTSAgent(1,2000),
        )

        current_player = 1

        players = {
            1: player1,
            -1: player2
        }

        board = Board()
        turn = 0

        opponentMove = None
            
        while True:
            turn += 1

            currentPlayer: Player = players[current_player]
            playerAgent = currentPlayer.agent

            ms = playerAgent.make_move(board, turn, opponentMove)


            board.do_move_sequence(ms,current_player)
            opponentMove = ms
            
            if board.get_winner() != 0 or turn == MAX_TURNS:
                break

            current_player = -current_player

        print(f"Process {p:>8d} Finished")
        return board.get_winner(), turn
    except Exception as e:
        file_name = f"process_error_{os.getpid()}"

        with open(f"{file_name}.txt", "a") as f:
            f.write(f"PID: {os.getpid()}\n")
            f.write(f"Exception: {str(e)}\n")
            f.write(traceback.format_exc())
            f.write("\n" + "="*60 + "\n")
        raise  # preserves original traceback

def init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

if __name__ == "__main__":
    NUM_CORES = 16

    red_wins = 0
    blu_wins = 0
    turn_outs = 0

    num_games = 1000


    print(f"Started at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    s = time.time_ns()
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = [executor.submit(play_game, i) for i in range(num_games)]
        results = [f.result() for f in futures]
    for r,_ in results:
        if r == 1:
            red_wins += 1
        elif r == -1:
            blu_wins += 1
        else:
            turn_outs += 1
    e = time.time_ns()
    print(f"Ended at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    
    print(f"Red Wins: {red_wins}")
    print(f"Blue Wins: {blu_wins}")
    print(f"Turn Outs: {turn_outs}")
    print(f"Time: {(e-s)/10**9}")
    
