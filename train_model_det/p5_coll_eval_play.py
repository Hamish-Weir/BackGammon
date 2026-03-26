import glob
import sys

def count_results(path):
    a0 = other = draw = 0

    for file in glob.glob(f"{path}/*.txt"):
        with open(file, "r") as f:
            last_line = f.readlines()[-1].strip()

        if last_line == "1":
            a0 += 1
        elif last_line == "-1":
            other += 1
        elif last_line == "0":
            draw += 1

    return a0, other, draw


M_R = count_results("data/eval_play_games_M_R")
M_B = count_results("data/eval_play_games_M_B")
R_R = count_results("data/eval_play_games_R_R")
R_B = count_results("data/eval_play_games_R_B")

output_path = f"data/b5_epoch_{sys.argv[1]}.txt"

with open(output_path, "w") as f:
    f.write("A0 (RED) vs MCTS (BLUE):")
    f.write(f"  A0 wins: {M_R[0]}")
    f.write(f"  MCTS wins: {M_R[1]}")
    f.write(f"  Turn/Time outs: {M_R[2]}\n")

    f.write("\nMCTS (RED) vs A0 (BLUE):")
    f.write(f"  A0 wins: {M_B[0]}")
    f.write(f"  MCTS wins: {M_B[1]}")
    f.write(f"  Turn/Time outs: {M_B[2]}\n")

    f.write("\nA0 (RED) vs RAND (BLUE):")
    f.write(f"  A0 wins: {R_R[0]}")
    f.write(f"  RAND wins: {R_R[1]}")
    f.write(f"  Turn/Time outs: {R_R[2]}\n")

    f.write("\nRAND (RED) vs A0 (BLUE):")
    f.write(f"  A0 wins: {R_B[0]}")
    f.write(f"  RAND wins: {R_B[1]}")
    f.write(f"  Turn/Time outs: {R_B[2]}\n")