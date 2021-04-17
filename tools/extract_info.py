import json

EXTRACT_DIR = "tuning_results/experiment42"

for i in range(12):
    repeat = 3
    best_score_sum = 0
    low_score = 1e20
    high_score = 0
    for j in range(repeat):
        with open(f"{EXTRACT_DIR}/test_gadqn_trial={i}_repeat={j}/params.json", "r") as f:
            results = json.load(f)
        best_score = results["Best Score"]
        best_score_sum += best_score
        low_score = best_score if best_score < low_score else low_score
        high_score = best_score if best_score > high_score else high_score
    avg = best_score_sum / repeat
    print(f"Trial: {i}, Avg best: {avg / 1e9}, Low: {low_score / 1e9}, High: {high_score / 1e9}")
