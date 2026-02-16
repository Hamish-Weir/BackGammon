
from concurrent.futures import ProcessPoolExecutor
import os
import multiprocessing as mp

from agents.RandomAgent import RandomAgent
from agents.MCTSAgent import MCTSAgent
from src.Player import Player
from src.Board import Board


def play_game(p):
    print(f"Process {p:>4d} Started")
    try:
        player1 = Player(
            "P1",
            MCTSAgent(1),
        )

        player2 = Player(
            "P1",
            MCTSAgent(-1),
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

            ms = playerAgent.make_move(board, opponentMove)

            board.do_move_sequence(ms,current_player)
            opponentMove = ms
            
            if board.get_winner() != 0:
                break

            current_player = -current_player

        print(f"Process {p:>4d} Finished")
        return board.get_winner(), turn
    except Exception as e:
        raise e


def init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

if __name__ == "__main__":
    NUM_CORES = 16

    red_wins = 0
    blu_wins = 0

    num_processes = 20
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = [executor.submit(play_game, i) for i in range(20)]
        results = [f.result() for f in futures]
    for r,_ in results:
        if r == 1:
            red_wins += 1
        elif r == -1:
            blu_wins += 1
        else:
            print(r)
            print(results)
            raise
    
    print(f"Red Wins: {red_wins}")
    print(f"Blue Wins: {blu_wins}")
    
