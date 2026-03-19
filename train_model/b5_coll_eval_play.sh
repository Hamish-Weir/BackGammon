#!/bin/bash --login

#SBATCH -p serial
#SBATCH -t 4
#SBATCH -o data/b5_eval_games.out


# A0 vs MCTS
    M_R_RED=0
    M_R_BLUE=0
    M_R_DRAW=0

    for f in data/eval_play_games_M_R/game_*.out; do
        result=$(tail -n 1 "$f")

        if [ "$result" = "1" ]; then
            ((M_R_RED++))
        elif [ "$result" = "-1" ]; then
            ((M_R_BLUE++))
        elif [ "$result" = "0" ]; then
            ((M_R_DRAW++))
        fi
    done

# MCTS vs A0
    M_B_RED=0
    M_B_BLUE=0
    M_B_DRAW=0

    for f in data/eval_play_games_M_B/game_*.out; do
        result=$(tail -n 1 "$f")

        if [ "$result" = "1" ]; then
            ((M_B_RED++))
        elif [ "$result" = "-1" ]; then
            ((M_B_BLUE++))
        elif [ "$result" = "0" ]; then
            ((M_B_DRAW++))
        fi
    done


# A0 vs RAND
    R_R_RED=0
    R_R_BLUE=0
    R_R_DRAW=0

    for f in data/eval_play_games_R_R/game_*.out; do
        result=$(tail -n 1 "$f")

        if [ "$result" = "1" ]; then
            ((R_R_RED++))
        elif [ "$result" = "-1" ]; then
            ((R_R_BLUE++))
        elif [ "$result" = "0" ]; then
            ((R_R_DRAW++))
        fi
    done

# RAND vs A0
    R_B_RED=0
    R_B_BLUE=0
    R_B_DRAW=0

    for f in data/eval_play_games_R_B/game_*.out; do
        result=$(tail -n 1 "$f")

        if [ "$result" = "1" ]; then
            ((R_B_RED++))
        elif [ "$result" = "-1" ]; then
            ((R_B_BLUE++))
        elif [ "$result" = "0" ]; then
            ((R_B_DRAW++))
        fi
    done

echo "A0 vs MCTS:"
echo "  Red wins: $M_R_RED"
echo "  Blue wins: $M_R_BLUE"
echo "  Turn/Time outs: $M_R_DRAW"

echo "MCTS vs A0:"
echo "  Red wins: $M_B_RED"
echo "  Blue wins: $M_B_BLUE"
echo "  Turn/Time outs: $M_B_DRAW"

echo "A0 vs RAND:"
echo "  Red wins: $R_R_RED"
echo "  Blue wins: $R_R_BLUE"
echo "  Turn/Time outs: $R_R_DRAW"

echo "RAND vs A0:"
echo "  Red wins: $R_B_RED"
echo "  Blue wins: $R_B_BLUE"
echo "  Turn/Time outs: $R_B_DRAW"

rm data/eval_play_games_M_R/*
rm data/eval_play_games_M_B/*
rm data/eval_play_games_R_R/*
rm data/eval_play_games_R_B/*
