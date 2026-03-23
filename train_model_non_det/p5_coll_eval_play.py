import glob

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


print(f"A0 (RED) vs MCTS (BLUE):")
print(f"  A0 wins: {M_R[0]}")
print(f"  MCTS wins: {M_R[1]}")
print(f"  Turn/Time outs: {M_R[2]}")

print(f"\nMCTS (RED) vs A0 (BLUE):")
print(f"  A0 wins: {M_B[0]}")
print(f"  MCTS wins: {M_B[1]}")
print(f"  Turn/Time outs: {M_B[2]}")

print(f"\nA0 (RED) vs RAND (BLUE):")
print(f"  A0 wins: {R_R[0]}")
print(f"  RAND wins: {R_R[1]}")
print(f"  Turn/Time outs: {R_R[2]}")

print(f"\nRAND (RED) vs A0 (BLUE):")
print(f"  A0 wins: {R_B[0]}")
print(f"  RAND wins: {R_B[1]}")
print(f"  Turn/Time outs: {R_B[2]}")