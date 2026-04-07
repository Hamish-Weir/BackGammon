import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from agents.A0Agent import A0Node
from networks.A0Network import A0Network
from src.Board import BOARD_SIZE, DIE_SIZE, HOME_SIZE, TOTAL_PLAYER_PIECES, Board


BEST_MODEL_PATH = f"models/{BOARD_SIZE}_{DIE_SIZE}_{HOME_SIZE}_{TOTAL_PLAYER_PIECES}_best_model.pth"
GAME_DATA_PATH = f"data/{BOARD_SIZE}_{DIE_SIZE}_{HOME_SIZE}_{TOTAL_PLAYER_PIECES}_dataset.pkl"

learning_rate = 0.0002
batch_size = 32
num_epochs = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# -------------------------
# Model
# -------------------------
best_model = A0Network()

try:
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    print("Loaded existing best model", flush=True)
except:
    best_model._initialize_weights()
    torch.save(best_model.state_dict(), BEST_MODEL_PATH)
    print("Training new model from scratch", flush=True)

best_model.to(device)

# -------------------------
# Dataset (memory efficient)
# -------------------------
def load_game_buffer(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        return pickle.load(f)

class GameDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        s, p, v = self.buffer[idx]
        return (
            s,
            torch.tensor(p, dtype=torch.float32),
            torch.tensor(v, dtype=torch.float32),
        )

game_buffer = load_game_buffer(GAME_DATA_PATH)
dataset = GameDataset(game_buffer)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------------------------
# Training
# -------------------------
optimizer = Adam(best_model.parameters(), lr=learning_rate)

print("Starting Training")

for epoch in range(num_epochs):
    best_model.train()
    total_loss = 0

    for s_batch, p_batch, v_batch in loader:
        s_batch = s_batch.to(device)
        p_batch = p_batch.to(device)
        v_batch = v_batch.to(device)

        v_pred, pol_pred = best_model(s_batch)

        v_pred = v_pred.squeeze(-1)
        pol_pred = F.log_softmax(pol_pred, dim=1)

        policy_loss = -(p_batch * pol_pred).sum(dim=1).mean()
        value_loss = F.mse_loss(v_pred, v_batch)

        loss = policy_loss + value_loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} loss: {total_loss / len(loader)}")

    # -------------------------
    # Evaluation (no grad!)
    # -------------------------
    best_model.eval()
    with torch.no_grad():
        b = Board()
        print("Starting Game State")

        for die1, die2 in [(i, j) for i in range(1, DIE_SIZE+1) for j in range(1, i+1)]:
            n = A0Node(b, die1, die2, 1)
            v = n.get_val_init_pri(best_model)
            p_dic = n.group_prior

            print(f"    Dice Roll: {(die1, die2)}")
            print(f"        Value: {v}")
            print(f"        Game State Priors:")
            for move, prob in p_dic.items():
                print(f"            {move}, {prob:.5f}")
            print()

        b = Board()
        l = b.get_legal_movesequences(1, 1, 1)
        b.do_move_sequence(l[0], 1)

        print("2nd Game State")

        for die1, die2 in [(i, j) for i in range(1, DIE_SIZE+1) for j in range(1, i+1)]:
            n = A0Node(b, die1, die2, -1)
            v = n.get_val_init_pri(best_model)
            p_dic = n.group_prior

            print(f"    Dice Roll: {(die1, die2)}")
            print(f"        Value: {v}")
            print(f"        Game State Priors:")
            for move, prob in p_dic.items():
                print(f"            {move}, {prob:.5f}")
            print()

print("Finished Training")

torch.save(best_model.state_dict(), BEST_MODEL_PATH)
print("New Best Model Saved")