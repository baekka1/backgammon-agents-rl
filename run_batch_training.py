# run_batch_training.py
import numpy as np
from td0_agent import init_weights, play_batch_td0

def main():
    weights = init_weights()
    result = play_batch_td0(weights,
                            batch_size=32,
                            num_batches=1000,
                            alpha=1e-3,
                            gamma=1.0)
    print("Finished batch training.")
    print("Total steps:", result["total_steps"])
    print("Total finished games:", result["total_finished_games"])
    print("Weights (first 10):", result["weights"][:10])

if __name__ == "__main__":
    main()
