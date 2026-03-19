#!/bin/bash --login

#SBATCH -p serial
#SBATCH -t 1
#SBATCH -o play_many.o%j

echo "Aggregating results for $v1 vs $v2"

RED=0
BLUE=0
DRAW=0

for f in logs/game_*.out; do
    result=$(tail -n 1 "$f")

    if [ "$result" = "1" ]; then
        ((RED++))
    elif [ "$result" = "-1" ]; then
        ((BLUE++))
    elif [ "$result" = "0" ]; then
        ((DRAW++))
    fi
done

echo "Red wins: $RED"
echo "Blue wins: $BLUE"
echo "Turn outs: $DRAW"

rm -f logs/*
