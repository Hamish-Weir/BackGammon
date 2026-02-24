# from networks.A0Network import A0Network
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import os

import torch
from networks.A0Network import A0Network
from agents.A0Agent import A0Agent, A0Node
# from agents.MCTSAgent import MCTSAgent
from src.Game import Game
from src.Board import Board

import time
import numpy as np
import sounddevice as sd

boards = [
    np.array([
        0,0,
        2,0,0,0,0,-2,
        0,0,
    ]),

    np.array([
        0,0,
        2,0,0,-2,0,0,
        0,0,
    ]),

    np.array([
        0,0,
        0,0,2,-2,0,0,
        0,0,
    ]),

    np.array([
        0,0,
        0,-2,2,0,0,0,
        0,0,
    ]),

    np.array([
        0,0,
        0,-2,0,0,2,0,
        0,0,
    ]),

    np.array([
        0,0,
        1,0,0,0,1,-2,
        0,0,
    ]),
    
    np.array([
        0,0,
        1,0,0,-2,1,0,
        0,0,
    ]),

    np.array([
        0,0,
        1,-1,0,0,1,-1,
        0,0,
    ]),

    np.array([
        0,0,
        1,-2,0,0,1,0,
        0,0,
    ]),
]


m = A0Network()
# m._initialize_weights()

m.load_state_dict(torch.load("models/best_model.pth",weights_only=True))

p = 1
b = Board()

for arr in boards:
    b.set(arr)

    # n = A0Node(b,p)
    e = A0Node.encode_board(b,1)

    with torch.inference_mode():
        v,p = m(e)

    v = v.item()
    p = p.squeeze(0).cpu().numpy()

    print(v)
