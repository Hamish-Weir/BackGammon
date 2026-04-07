#!/bin/bash --login

#SBATCH -p serial
#SBATCH -t 8
#SBATCH -o data/b5_coll_eval_play.out

python -u p5_coll_eval_play.py $epoch

# rm data/eval_play_games_M_R/*
# rm data/eval_play_games_M_B/*
# rm data/eval_play_games_R_R/*
# rm data/eval_play_games_R_B/*