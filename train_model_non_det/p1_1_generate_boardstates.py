import pickle
from typing import Set

import numpy as np

from src.Board import BOARD_SIZE, DIE_SIZE, HOME_SIZE, TOTAL_PLAYER_PIECES, Board

LEGAL_BOARDS_PATH = f"data/{BOARD_SIZE}_{DIE_SIZE}_{HOME_SIZE}_{TOTAL_PLAYER_PIECES}_boardstates.pkl"

def collect_states_iterative(b:Board, depth, player):
    boardSet = set()
    boardlist = []

    new_state = tuple((tuple(b.get().flatten()),player))
    if new_state not in boardSet:
        boardSet.add(new_state)
        boardlist.append((b.get(),player))


    stack = [(b.get(), depth, player)]

    while stack:
        state, d, p = stack.pop()
        b.set(state)

        

        if d == 0:
            continue

        lms = []
        for d1 in range(1,DIE_SIZE+1):
            for d2 in range(d1,DIE_SIZE+1):
                lms.extend(b.get_legal_movesequences(d1,d2,p))

        for ms in lms:
            b.set(state)
            b.do_move_sequence(ms, p)

            if b.get_winner() != 0:
                continue

            new_state = tuple((tuple(b.get().flatten()),player))
            if new_state not in boardSet:
                boardSet.add(new_state)
                boardlist.append((b.get(),p))
                stack.append((b.get(), d - 1, -p))

    return boardlist

b = Board()

boardlist = collect_states_iterative(b,3,1)

with open(LEGAL_BOARDS_PATH, "wb") as f:
    pickle.dump(boardlist, f)
