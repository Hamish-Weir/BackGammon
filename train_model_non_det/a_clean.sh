#!/bin/bash

#SBATCH -p  serial
#SBATCH -t 10
#SBATCH -o /dev/null
#SBATCH -e /dev/null

rm data/eval_play_games_M_R/*
rm data/eval_play_games_M_B/*
rm data/eval_play_games_R_R/*
rm data/eval_play_games_R_B/*
rm data/self_play_games/*