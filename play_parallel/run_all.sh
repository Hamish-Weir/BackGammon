#!/bin/bash

v1="A0"
v2="MCTS"

# "MCTS", "A0", "RAND"

mkdir -p logs
rm -f logs/*

echo "Red Agent: $v1"
echo "Blu Agent: $v2"

# Submit game array job
games=$(sbatch --export=ALL,v1=$v1,v2=$v2 play_game_array.sh | awk '{print $4}')
echo "Submitted game array job with ID: $games"

# Submit aggregation job
agg=$(sbatch --export=ALL,v1=$v1,v2=$v2 --dependency=afterok:$games aggregate.sh | awk '{print $4}')
echo "Submitted aggregation job with ID: $agg (depends on $games)"

echo "Pipeline submitted successfully!"
