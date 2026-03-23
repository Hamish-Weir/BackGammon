#!/bin/bash

#SBATCH -p serial
#SBATCH -t 10
#SBATCH -o data/b0_train_cycle.out

mkdir -p data
mkdir -p data/self_play_games/
mkdir -p data/eval_play_games_M_R/
mkdir -p data/eval_play_games_M_B/
mkdir -p data/eval_play_games_R_R/
mkdir -p data/eval_play_games_R_B/

# rm -f data/self_play_games/*
rm -f data/eval_play_games_M_R/*
rm -f data/eval_play_games_M_B/*
rm -f data/eval_play_games_R_R/*
rm -f data/eval_play_games_R_B/*

epoch=1

# First cycle
play_self=$(sbatch --export=ALL,epoch=$epoch b1_play_self_play_array.sh | awk '{print $4}')
coll_self=$(sbatch --dependency=afterok:$play_self b2_coll_self_play.sh | awk '{print $4}')
train=$(sbatch --dependency=afterok:$coll_self b3_train_model.sh | awk '{print $4}')

play_eval_M_R=$(sbatch --dependency=afterok:$train b4_1_play_eval_M_R.sh | awk '{print $4}')
play_eval_M_B=$(sbatch --dependency=afterok:$train b4_2_play_eval_M_B.sh | awk '{print $4}')
play_eval_R_R=$(sbatch --dependency=afterok:$train b4_3_play_eval_R_R.sh | awk '{print $4}')
play_eval_R_B=$(sbatch --dependency=afterok:$train b4_4_play_eval_R_B.sh | awk '{print $4}')

coll_eval=$(sbatch --export=ALL,epoch=$epoch \
  --dependency=afterok:$play_eval_M_R:$play_eval_M_B:$play_eval_R_R:$play_eval_R_B \
  b5_coll_eval_play.sh | awk '{print $4}')

# Loop
for epoch in {2..20}; do
    play_self=$(sbatch --export=ALL,epoch=$epoch --dependency=afterok:$coll_eval b1_play_self_play_array.sh | awk '{print $4}')
    coll_self=$(sbatch --dependency=afterok:$play_self b2_coll_self_play.sh | awk '{print $4}')
    train=$(sbatch --dependency=afterok:$coll_self b3_train_model.sh | awk '{print $4}')

    play_eval_M_R=$(sbatch --dependency=afterok:$train b4_1_play_eval_M_R.sh | awk '{print $4}')
    play_eval_M_B=$(sbatch --dependency=afterok:$train b4_2_play_eval_M_B.sh | awk '{print $4}')
    play_eval_R_R=$(sbatch --dependency=afterok:$train b4_3_play_eval_R_R.sh | awk '{print $4}')
    play_eval_R_B=$(sbatch --dependency=afterok:$train b4_4_play_eval_R_B.sh | awk '{print $4}')

    coll_eval=$(sbatch --export=ALL,epoch=$epoch \
      --dependency=afterok:$play_eval_M_R:$play_eval_M_B:$play_eval_R_R:$play_eval_R_B \
      b5_coll_eval_play.sh | awk '{print $4}')

    echo "Epoch $epoch submitted"
done

echo "Pipeline submitted successfully!"