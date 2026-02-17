# from networks.A0Network import A0Network
import torch
from networks.A0Network import A0Network
from agents.A0Agent import A0Agent, A0Node
# from agents.MCTSAgent import MCTSAgent
from src.Game import Game
from src.Board import Board

import time
import numpy as np


b = Board()
arr = np.array([
    0,0,
    2,0,0,0,0,-2,
    0,0,
])

b.set(arr)
a = A0Agent(
    1,
    model_path="models/best_model.pth",
    simulations=500,
    c_puct=1)

ms = a.make_move(b)

print(a.get_rollout())

C0 = a.root                        
# # C1 = list(C0.children.values())[3]
# # C2 = list(C1.children.values())[0]
# # C3 = list(C2.children.values())[1]
# # C4 = list(C3.children.values())[1]

# # C0  1,3:1,3
# # C1 6,4:6,4
# # C2 

N = C0

print("Parent:")
print(N)
print("Children")
for i, c in enumerate(N.children.values()):
    print(f"{i}, {c}",end="")