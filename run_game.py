import numpy as np
from td0_agent import play_one_game_td0, init_weights

def main():
    weights = init_weights()

    # Play 1 game and print the final outcome + number of moves
    reward, steps = play_one_game_td0(weights)

    print("Game finished.")
    print("Final TD(0) reward from WHITE perspective:", reward)
    print("Game length (moves):", steps)
    print("Updated weights (first 10):", weights[:10])

if __name__ == "__main__":
    main()
