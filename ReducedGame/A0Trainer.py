from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import multiprocessing as mp
import os
import pickle
import traceback

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.Board import BOARD_END, BOARD_SIZE, BOARD_START, DIE_SIZE, GAME_SIZE, HOME_SIZE, TOTAL_PLAYER_PIECES, Board
from agents.A0Agent import A0Agent, A0Node
from networks.A0Network import A0Network

MAX_GAME_LENGTH = 15

BEST_MODEL_PATH = f"models/{BOARD_SIZE}_{DIE_SIZE}_{HOME_SIZE}_{TOTAL_PLAYER_PIECES}_best_model.pth"
TEMP_MODEL_PATH = f"models/{BOARD_SIZE}_{DIE_SIZE}_{HOME_SIZE}_{TOTAL_PLAYER_PIECES}_temp_model.pth"
GAME_DATA_PATH =  f"models/{BOARD_SIZE}_{DIE_SIZE}_{HOME_SIZE}_{TOTAL_PLAYER_PIECES}_dataset.pkl"

TRAIN_SIMS = 400
TRAIN_GAMES = 50
TRAIN_C_PUCT = 1
TRAIN_DIRICHLET_ALPHA = 0.1
TRAIN_DIRICHLET_EPSILON = 0.33
TRAIN_TEMPERATURE = 1
TRAIN_TEMPREATURE_PLY = 6

EVAL_SIMS = 1
EVAL_GAMES = 50
EVAL_C_PUCT = 1
EVAL_DIRICHLET_ALPHA = 0.1
EVAL_DIRICHLET_EPSILON = 0.25
EVAL_TEMPERATURE = 0
EVAL_TEMPREATURE_PLY = 0
CPU_COUNT = mp.cpu_count()

class A0Trainer:
    def __init__(
        self,
        learning_rate       = 0.002,
        batch_size          = 32,
        num_epochs          = 10,
        device              = None,
        number_of_cores     = CPU_COUNT,
        best_model_path     = BEST_MODEL_PATH,
        temp_model_path     = TEMP_MODEL_PATH,
        deque_path          = GAME_DATA_PATH
    ):
        self.core_no = min(number_of_cores,CPU_COUNT-1)

        self.value_c        = 2
        self.learning_rate  = learning_rate
        self.batch_size     = batch_size
        self.num_epochs     = num_epochs

        self.deque_path     = deque_path
        self.deque_size     = TRAIN_GAMES * 5
        self.game_buffer    = self._load_deque()

        if device:
            try:
                self.device             = torch.device(device)
            except:
                print(f"No Device '{device}', using default")
                self.device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if torch.cuda.is_available():
                    print("Running on: GPU")
                else:
                    print("Running on: CPU")
        else:
            self.device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                print("Running on: GPU")
            else:
                print("Running on: CPU")

        self.best_model_path = best_model_path
        self.temp_model_path = temp_model_path
        self.best_model = A0Network()
        self.temp_model = A0Network()

        # Load best model if exists
        try:
            self.best_model.load_state_dict(torch.load(self.best_model_path))
            self.temp_model.load_state_dict(torch.load(self.best_model_path))
            print("Loaded existing best model", flush=True)
        except:
            self.best_model._initialize_weights()
            torch.save(self.best_model.state_dict(), self.best_model_path)
            self.temp_model.load_state_dict(self.best_model.state_dict())
            print("Training new model from scratch", flush=True)

        print(f"Best Model Stored at: {self.best_model_path}")
        print(f"Temp Model Stored at: {self.temp_model_path}")
        print(f"Game Data Stored at: {self.deque_path}")
        print(f"Playing games on: {self.core_no} Cores")
        print()

    def train(self,iterations = 50):
        for i in range(iterations):
            print()
            print(f"Iteration {i+1} started at: {time_str()}")

            print(f"Model Data-Generation started at: {time_str()}")
            self._play_step()

            print(f"Model Training started at: {time_str()}")
            self._train_step()

            print(f"Model Evaluation started at: {time_str()}")
            self._evaluate_step()

            print(f"Iteration {i+1} Ended at: {time_str()}")

            self._show_progress()

    def _play_step(self):

        game_data = generate_dataset(self.best_model_path)

        self.game_buffer.extend(game_data)
        self._save_deque() 

    def _train_step(self):
        self.temp_model.load_state_dict(torch.load(self.best_model_path))
        self.temp_model.to(self.device)
        self.temp_model.train()
        flat_dataset = [pos for game in self.game_buffer for pos in game]

        optimizer = Adam(
            self.temp_model.parameters(),
            lr=self.learning_rate,  # initial learning rate
        )

        self.temp_model.to(self.device)
        self.temp_model.train()
        
        states = torch.stack([s for (s, _, _) in flat_dataset]).to(self.device)
        policies = torch.tensor(np.array([p for (_, p, _) in flat_dataset]), dtype=torch.float).to(self.device)
        values = torch.tensor(np.array([v for (_, _, v) in flat_dataset]), dtype=torch.float).to(self.device)

        for epoch in range(self.num_epochs):
            total_loss = 0
            perm = torch.randperm(len(states))
            states = states[perm]
            policies = policies[perm]
            values = values[perm]

            for start in range(0, len(states), self.batch_size):
                s_batch = states[start:start + self.batch_size]
                p_batch = policies[start:start + self.batch_size]
                v_batch = values[start:start + self.batch_size]

                v_pred, pol_pred = self.temp_model(s_batch)

                v_pred = v_pred.squeeze(-1)
                pol_pred = F.log_softmax(pol_pred, dim=1)

                # Loss = policy loss + value loss + L2 regularization
                policy_loss = -(p_batch * pol_pred).sum(dim=1).mean()
                value_loss = F.mse_loss(v_pred, v_batch)

                # for vp,vb in zip(v_pred,v_batch):
                #     print(int(vp),int(vb))

                loss = policy_loss + self.value_c * value_loss
                total_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print(f"Epoch {epoch+1} loss: {total_loss}")

        self.temp_model.eval()

        return

    def _evaluate_step(self):
        torch.save(self.temp_model.state_dict(), self.temp_model_path)

        # Red_winrate, Blue_winrate = evaluation_games(self.best_model_path,self.temp_model_path)
        # print(f"(Red win rate = {Red_winrate}), (Blue win rate = {Blue_winrate})")

        # if winrate > 0.55:
        torch.save(self.temp_model.state_dict(), self.best_model_path)
        print(f"New model saved")
        # else:
        #     print(f"New model discarded (Winrate = {winrate})")

    def _load_deque(self):
        """Safely load a deque"""
        
        if not os.path.exists(self.deque_path):
            print(f"{self.deque_path} not found. Creating a new empty deque.")
            return deque(maxlen=self.deque_size)

        try:
            with open(self.deque_path, "rb") as f:
                obj = pickle.load(f)

            if not isinstance(obj, deque):
                print(f"Warning: {self.deque_path} did not contain a deque. Creating a new one.")
                return deque(maxlen=self.deque_size)
            
            print("Loaded Existing Game Data")

            return obj

        except Exception as e:
            print(f"Error loading {self.deque_path}: {e}")
            print("Creating a new empty deque.")
            return deque(maxlen=self.deque_size)
      
    def _save_deque(self):
        """Safely save a deque to disk."""
        with open(self.deque_path, "wb") as f:
            pickle.dump(self.game_buffer, f)

    def _show_progress(self):
        print()

        self.best_model.load_state_dict(torch.load(self.best_model_path))

        b = Board()

        n = A0Node(b,1)
        v = n.get_val_init_pri(self.best_model)
        p_dic = n.child_prior

        print(f"Starting Game State Value: {v}")
        print(f"Starting Game State Priors:")
        for i in list(p_dic.items()):
            print(f"{i[0]}, {i[1]:.5f}")

        print()

        b = Board()

        l = b.get_legal_movesequences(1)
        b.do_move_sequence(l[1],1)
        
        # arr = b.get()
        # arr[BOARD_START] = 0
        # arr[BOARD_START+1] = 2
            
        # b.set(arr)

        n = A0Node(b,-1)
        v = n.get_val_init_pri(self.best_model)
        p_dic = n.child_prior

        print(f"Starting Game State Value: {v}")
        print(f"Starting Game State Priors:")
        for i in list(p_dic.items()):
            print(f"{i[0]}, {i[1]:.5f}")

        print()


def time_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def self_play_game(
        model_path,
        simulations=TRAIN_SIMS,
        c_puct=TRAIN_C_PUCT,
        dir_a=TRAIN_DIRICHLET_ALPHA,
        dir_e=TRAIN_DIRICHLET_EPSILON,
        temp=TRAIN_TEMPERATURE,
        temp_ply=TRAIN_TEMPREATURE_PLY,
    ):
    try:
        agent1=A0Agent(1,model_path,simulations,c_puct,True,dir_a,dir_e,temp,temp_ply)
        agent2=A0Agent(-1,model_path,simulations,c_puct,True,dir_a,dir_e,temp,temp_ply)

        players = {
            1: agent1,
            -1: agent2,
        }

        current_player = 1
        board = Board()
        turn = 0

        opponentMove = None

        game_history = []

        while True:
            turn += 1

            playerAgent:A0Agent = players[current_player]
            
            ms = playerAgent.make_move(board,turn, opponentMove)

            tensor = A0Node.encode_board(board,current_player)
            prior = playerAgent.get_rollout()
            playee = current_player

            state = (tensor,prior,playee)
            
            game_history.append(state)
            
            board.do_move_sequence(ms,current_player)

            if board.get_winner() != 0 or turn > MAX_GAME_LENGTH:
                break
            
            current_player = -current_player
        
        winner = board.get_winner() # win, loss, or turn out

        for i, (s, p, pl) in enumerate(game_history):
            if pl == winner:
                value = 1 
            elif -pl == winner:
                value = -1
            else:
                value = 0

            game_history[i] = (s, p, value)

        return game_history
    
    except Exception as e:
        # write full traceback to a per-process file so you can inspect it
        pid = os.getpid()
        with open(f"worker_error_{pid}.log", "w") as f:
            f.write("Exception in worker:\n")
            traceback.print_exc(file=f)
        raise

def eval_play_game(
        parity,
        best_model_path,
        temp_model_path,
        simulations=EVAL_SIMS,
        c_puct=EVAL_C_PUCT,
        dir_a=EVAL_DIRICHLET_ALPHA,
        dir_e=EVAL_DIRICHLET_EPSILON,
        temp=EVAL_TEMPERATURE,
        temp_ply=EVAL_TEMPREATURE_PLY,
):
    try:
        if parity == 0:
            agent1=A0Agent(1,temp_model_path,simulations,c_puct,True,dir_a,dir_e,temp,temp_ply) # New as Red
            agent2=A0Agent(-1,best_model_path,simulations,c_puct,True,dir_a,dir_e,temp,temp_ply)
        else:
            agent1=A0Agent(1,best_model_path,simulations,c_puct,True,dir_a,dir_e,temp,temp_ply)
            agent2=A0Agent(-1,temp_model_path,simulations,c_puct,True,dir_a,dir_e,temp,temp_ply) # New as Blue

        players = {
            1: agent1,
            -1: agent2,
        }

        current_player = 1
        board = Board()
        turn = 0

        opponentMove = None

        while True:
            turn += 1

            playerAgent:A0Agent = players[current_player]
            
            ms = playerAgent.make_move(board,turn, opponentMove)

            board.do_move_sequence(ms,current_player)

            if board.get_winner() != 0 or turn > MAX_GAME_LENGTH:
                break
            
            current_player = -current_player
        
        winner = board.get_winner() # win, loss, or turn out

        if winner == 1 and parity == 0:
            return 1
        elif winner == -1 and parity == 1:
            return 1
        else:
            return -1

    except Exception as e:
        # write full traceback to a per-process file so you can inspect it
        pid = os.getpid()
        with open(f"worker_error_{pid}.log", "w") as f:
            f.write("Exception in worker:\n")
            traceback.print_exc(file=f)
        raise

def worker_init():
    # Called once per worker process
    # Limit threads used by numpy/pytorch/...
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

def generate_dataset(model_path):

    # explicit multiprocessing context 
    ctx = mp.get_context("spawn")

    dataset = []
    finished = 0
    with ProcessPoolExecutor(max_workers=CPU_COUNT, mp_context=ctx, initializer=worker_init) as ex:
        futures = [ex.submit(self_play_game,model_path) for _ in range(TRAIN_GAMES)]
        for future in as_completed(futures):
            # If a worker raised, .result() will re-raise that exception here.
            game = future.result()
            dataset.append(game)

            finished += 1
            print(f"{finished}/{TRAIN_GAMES}, ",end="",flush=True)

    red_wins = 0
    blu_wins = 0
    non_wins = 0

    for game in dataset:
        w = game[0][2]
        if w == 1:
            red_wins += 1
        if w == -1:
            blu_wins += 1
        if w == 0:
            non_wins += 1

    print(f"\nRed Wins: {red_wins}, Blue Wins {blu_wins}, Draws: {non_wins}")

    return dataset

def evaluation_games(best_model_path,temp_model_path):
    
    # explicit multiprocessing context 
    ctx = mp.get_context("spawn")

    Red_Wins = 0
    Blue_Wins = 0

    with ProcessPoolExecutor(max_workers=CPU_COUNT, mp_context=ctx, initializer=worker_init) as ex:
        futures = [ex.submit(eval_play_game,0,best_model_path,temp_model_path) for i in range(EVAL_GAMES//2)]
        for future in as_completed(futures):
            # If a worker raised, .result() will re-raise that exception here.
            winner = future.result()
            if winner == 1:
                Red_Wins += 1

    with ProcessPoolExecutor(max_workers=CPU_COUNT, mp_context=ctx, initializer=worker_init) as ex:
        futures = [ex.submit(eval_play_game,1,best_model_path,temp_model_path) for i in range(EVAL_GAMES//2)]
        for future in as_completed(futures):
            # If a worker raised, .result() will re-raise that exception here.
            winner = future.result()
            if winner == 1:
                Blue_Wins+=1

    R_winrate = Red_Wins/(TRAIN_GAMES//2)
    B_winrate = Blue_Wins/(TRAIN_GAMES//2)

    return R_winrate,B_winrate

def play_finish_sound():
    import sounddevice as sd

    sample_rate = 44100

    t1 = np.linspace(0, 0.4, int(sample_rate * 0.4), False)
    tone1 = np.sin(2 * np.pi * 1000 * t1)

    t2 = np.linspace(0, 0.5, int(sample_rate * 0.5), False)
    tone2 = np.sin(2 * np.pi * 2000 * t2)

    tone = np.concatenate([tone1, tone2,tone1, tone2,tone1, tone2,tone1, tone2])

    sd.play(tone, sample_rate)
    sd.wait()

if __name__ == '__main__':
    print(f"Start Time: {datetime.now()}")
    
    trainer = A0Trainer()

    trainer.train(iterations=100)

    play_finish_sound()

    # data = generate_dataset("models/successful_model.pth")

    # for game in data:
    #     for state in game:
    #         tensor, pol, val = state
    #         print(f"Tensor:\n{tensor}")
    #         print(f"Prior: \n{pol}")
    #         print(f"Value: {val}")
    #         print()



# A0Model: took 1 hr to train, 20 itterations