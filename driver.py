import argparse
import json
import os


def driver():
    """
    TVM tools command line driver.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", required=True, help="Select method type tune/benchmark/experiment")
    parser.add_argument("-c", "--config", required=True, help="JSON configuration for specified method")
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config) as f:
            json_config = json.load(f)
    else:
        json_config = json.loads(args.config)

    # run specified method
    if args.method == "tune":
        from tools.tune_model import tune_models
        tune_models(json_config)
    elif args.method == "benchmark":
        from tools.benchmark_model import benchmark_models
        benchmark_models(json_config)
    elif args.method == "experiment":
        from tools.experiments import run_experiments
        run_experiments(json_config)
    else:
        raise ValueError("Specified method not recognised. Use tune/benchmark/experiment.")


if __name__ == "__main__":
    driver()
