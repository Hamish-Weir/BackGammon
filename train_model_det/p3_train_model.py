import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from agents.A0Agent import A0Node
from networks.A0Network import A0Network
from src.Board import BOARD_SIZE, DIE_SIZE, HOME_SIZE, TOTAL_PLAYER_PIECES, Board


BEST_MODEL_PATH = f"models/{BOARD_SIZE}_{DIE_SIZE}_{HOME_SIZE}_{TOTAL_PLAYER_PIECES}_best_model.pth"
GAME_DATA_PATH =  f"data/{BOARD_SIZE}_{DIE_SIZE}_{HOME_SIZE}_{TOTAL_PLAYER_PIECES}_dataset.pkl"

learning_rate       = 0.00002
batch_size          = 64
num_epochs          = 2


# Load Model and Data
device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Running on: GPU")
else:
    print("Running on: CPU")

best_model = A0Network()

try:
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH))
    print("Loaded existing best model", flush=True)
except:
    best_model._initialize_weights()
    torch.save(best_model.state_dict(), BEST_MODEL_PATH)
    print("Training new model from scratch", flush=True)

def load_game_buffer(Dataset_Path):
    if not os.path.exists(Dataset_Path):
        print(f"{Dataset_Path} not found.")
        raise
    else:
        with open(Dataset_Path, "rb") as f:
            dataset = pickle.load(f)
        return dataset
    
game_buffer = load_game_buffer(GAME_DATA_PATH)

# Train Model
optimizer = Adam(
    best_model.parameters(),
    lr=learning_rate,  # initial learning rate
)


best_model.train()
        
states = torch.stack([s for (s, _, _) in game_buffer]).to(device)
policies = torch.tensor(np.array([p for (_, p, _) in game_buffer]), dtype=torch.float).to(device)
values = torch.tensor(np.array([v for (_, _, v) in game_buffer]), dtype=torch.float).to(device)

print("Starting Training")
for epoch in range(num_epochs):
    best_model.to(device)
    total_loss = 0
    perm = torch.randperm(len(states))
    states = states[perm]
    policies = policies[perm]
    values = values[perm]

    for start in range(0, len(states), batch_size):
        s_batch = states[start:start + batch_size]
        p_batch = policies[start:start + batch_size]
        v_batch = values[start:start + batch_size]

        v_pred, pol_pred = best_model(s_batch)

        v_pred = v_pred.squeeze(-1)
        pol_pred = F.log_softmax(pol_pred, dim=1)

        # Loss = policy loss + value loss + L2 regularization
        policy_loss = -(p_batch * pol_pred).sum(dim=1).mean()
        value_loss = F.mse_loss(v_pred, v_batch)

        # for vp,vb in zip(v_pred,v_batch):
        #     print(int(vp),int(vb))

        loss = policy_loss + value_loss
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs} loss: {total_loss/len(states)}")
    
    print()
    best_model.to(torch.device("cpu"))
    b = Board()
    print(f"Starting Game State")
    n = A0Node(b,1)
    v = n.get_val_init_pri(best_model)
    p_dic = n.child_prior

    print(f"        Value: {v}")
    print(f"        Game State Priors:")
    for i in list(p_dic.items()):
        print(f"            {i[0]}, {i[1]:.5f}")
    print()


    b = Board()

    l = b.get_legal_movesequences(1)
    b.do_move_sequence(l[0],1)

    print(f"2nd Game State")
    n = A0Node(b,-1)
    v = n.get_val_init_pri(best_model)
    p_dic = n.child_prior

    print(f"        Value: {v}")
    print(f"        Game State Priors:")
    for i in list(p_dic.items()):
        print(f"            {i[0]}, {i[1]:.5f}")
    print()


print("Finished Training")

torch.save(best_model.state_dict(), BEST_MODEL_PATH)
print("New Best Model Saved")

