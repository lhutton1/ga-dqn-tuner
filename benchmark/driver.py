import argparse
import json


def driver():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", required=True, help="Select method type tune/benchmark")
    parser.add_argument("-c", "--config", required=True, help="JSON configuration for specified method")
    args = parser.parse_args()
    json_config = json.loads(args.config)

    # run specified method
    if args.method == "tune":
        from tune_model import tune_models
        tune_models(json_config)
    elif args.method == "benchmark":
        from benchmark_model import benchmark_models
        benchmark_models(json_config)
    elif args.method == "rl_optimize":
        from trial_hyperparameters import trial_parameters
        trial_parameters(json_config)
    else:
        raise ValueError("Specified method not recognised. Use tune/benchmark.")


if __name__ == "__main__":
    driver()
