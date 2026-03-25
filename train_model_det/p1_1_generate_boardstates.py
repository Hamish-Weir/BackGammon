import pickle
from typing import Set

import numpy as np

from src.Board import BOARD_SIZE, DIE_SIZE, HOME_SIZE, TOTAL_PLAYER_PIECES, Board

LEGAL_BOARDS_PATH = f"data/{BOARD_SIZE}_{DIE_SIZE}_{HOME_SIZE}_{TOTAL_PLAYER_PIECES}_boardstates.pkl"

def collect_states_iterative(b, depth, player):
    boardSet = set()
    boardlist = []

    new_state = tuple(b.get().flatten())
    if new_state not in boardSet:
        boardSet.add(new_state)
        boardlist.append(b.get())


    stack = [(b.get(), depth, player)]

    while stack:
        state, d, p = stack.pop()
        b.set(state)

        if d == 0:
            continue

        lms = b.get_legal_movesequences(p)

        for ms in lms:
            b.set(state)
            b.do_move_sequence(ms, p)

            new_state = tuple(b.get().flatten())
            if new_state not in boardSet:
                boardSet.add(new_state)
                boardlist.append((b.get(),p))
                stack.append((b.get(), d - 1, -p))

    return boardlist

b = Board()

boardlist = collect_states_iterative(b,10,1)

with open(LEGAL_BOARDS_PATH, "wb") as f:
    pickle.dump(boardlist, f)
