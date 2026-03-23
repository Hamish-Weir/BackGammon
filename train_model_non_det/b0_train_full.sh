#!/bin/bash

#SBATCH -p  serial
#SBATCH -t 1
#SBATCH -o data/b0_train_cycle.out

# TODO: use cycle number for cleaning and logging

# Make Data
mkdir data

# Train Data
mkdir -p data/self_play_games/

# TODO Change later
rm data/self_play_games/*

# Eval Data
mkdir -p data/eval_play_games_M_R/
mkdir -p data/eval_play_games_M_B/
mkdir -p data/eval_play_games_R_R/
mkdir -p data/eval_play_games_R_B/

# TODO Change later
rm data/eval_play_games_M_R/*
rm data/eval_play_games_M_B/*
rm data/eval_play_games_R_R/*
rm data/eval_play_games_R_B/*



# Self Play Games
    # play_self_play.py ✅

# Collect Game Data
    # coll_self_play.py ✅
        # output pkl file?

# Train Model
    # train_model.py 

# Play Eval Games
    # play_eval_mcts_red.py
    # play_eval_mcts_blue.py
    # play_eval_rand_red.py
    # play_eval_rand_blue.py

# Collect Eval Games
    # coll_eval_mcts_red.py
    # coll_eval_mcts_blue.py
    # coll_eval_rand_red.py
    # coll_eval_rand_blue.py


# Self Play Games
    play_self=$(sbatch b1_play_self_play_array.sh | awk '{print $4}')
    echo "Submitted Game Array job with ID: $play_self"

# Collect Self Data
    coll_self=$(sbatch --dependency=afterok:$play_self b2_coll_self_play.sh | awk '{print $4}')
    echo "Submitted Collection job with ID: $coll_self (depends on $play_self)"

# Train Model
    train=$(sbatch --dependency=afterok:$coll_self b3_train_model.sh | awk '{print $4}')
    echo "Submitted Train job with ID: $train (depends on $coll_self)"

# Play Eval Games
    play_eval_M_R=$(sbatch --dependency=afterok:$train b4_1_play_eval_M_R.sh | awk '{print $4}')
    echo "Submitted Game Array job with ID: $play_eval_M_R (depends on $train)"

    play_eval_M_B=$(sbatch --dependency=afterok:$train b4_2_play_eval_M_B.sh | awk '{print $4}')
    echo "Submitted Game Array job with ID: $play_eval_M_B (depends on $train)"

    play_eval_R_R=$(sbatch --dependency=afterok:$train b4_3_play_eval_R_R.sh | awk '{print $4}')
    echo "Submitted Game Array job with ID: $play_eval_R_R (depends on $train)"

    play_eval_R_B=$(sbatch --dependency=afterok:$train b4_4_play_eval_R_B.sh | awk '{print $4}')
    echo "Submitted Game Array job with ID: $play_eval_R_B (depends on $train)"


# Collect Eval Data
    coll_eval=$(sbatch --dependency=afterok:$play_eval_M_R:$play_eval_M_B:$play_eval_R_R:$play_eval_R_B b5_coll_eval_play.sh | awk '{print $4}')
    echo "Submitted Collection job with ID: $coll_eval (depends on $play_eval_M_R, $play_eval_M_B, $play_eval_R_R, $play_eval_R_B)"


echo "Pipeline submitted successfully!"
