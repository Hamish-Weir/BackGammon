#!/bin/bash --login

#SBATCH -p serial
#SBATCH -t 8
#SBATCH -o /dev/null
#SBATCH -e /dev/null

play_eval_M_R=$(sbatch b4_1_play_eval_M_R.sh | awk '{print $4}')
play_eval_M_B=$(sbatch b4_2_play_eval_M_B.sh | awk '{print $4}')
play_eval_R_R=$(sbatch b4_3_play_eval_R_R.sh | awk '{print $4}')
play_eval_R_B=$(sbatch b4_4_play_eval_R_B.sh | awk '{print $4}')

coll_eval=$(sbatch --export=ALL,epoch=$(9999) \
    --dependency=afterok:$play_eval_M_R:$play_eval_M_B:$play_eval_R_R:$play_eval_R_B \
    b5_coll_eval_play.sh | awk '{print $4}')

echo "Submitted Evaluation Games"