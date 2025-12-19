# Reinforcement Learning Final Project
Authors: Katie Baek and Josh Gorniak

# Set Up
First, set up a virtual environment: `python3 -m venv .venv`  
Then, activate the environment: `source .venv/bin/activate`  
Download of the necessary libraries: `pip install -r requirements.txt`

# Agent 1
**Josh Gorniak**

This agent is a TD(0) linear value-function agent that learns a handcrafted evaluation function and selects moves using 2-ply search. The value function is a linear model over domain-specific backgammon features, including raw board encoding, blot and prime structure, pip counts, and race/contact indicators (as specified in the assignment instructions). Training is performed via vectorized self-play, where multiple games are run in parallel and a shared weight vector is updated using averaged TD(0) targets. I developed the code iteratively, with my initial tests first encompassing a single game run linearly (`play_one_game_td0`). This agent serves as a classical baseline for comparison with the neural TD($\lambda$) and PPO agents.

Running `python td0_agent.py` begins batch training and returns the first 10 weights.

# Agent 2
**Katie Baek**

This agent is a TD($\lambda$) program that learns a neural value function and uses 2-ply search to chose moves. I decided to create an agent class called `BackgammonAgent`, which contains the model parameters for the `BackgammonValueNet` as well as all functions which are related to doing the training of the model. The environment state is as described in the project, with 15 feature planes and 6 auxillery features. 

To train the model, I first initalized a `BackgammonAgent` agent. Then for each episode, I created a batch of games and for each game, select an action via 2-ply search, applies the moves in the engine, computes rewards and bootstrap values, and updates the value network using semi-gradient TD($\lambda$) with eligibility traces.  

Running `python tdlambda.py` trains an agent on 5 episodes with 5 games, and then does one real game with the trained agent.

# Agent 3
**Josh Gorniak x Katie Baek**

We implemented Agent 3 as a self-play training system that learns to play backgammon by running many games in parallel and training a single neural network to both evaluate positions and choose moves. We convert every game state into a consistent current-player view, encode it using the assignment’s board and auxiliary features, and use the policy network to sample moves step by step until a legal move is formed. We then apply those moves in the engine, collect rewards and value estimates, and update the network using PPO so that better decisions become more likely over time.

The command accepts three flags: `python ppo_agent.py --num_envs 8 --steps_per_rollout 16 --total_updates=100`

We have also attached our first attempt at programming a PPO agent, so that our initial efforts were not in vain: `ppo_agent_deprecated.py`



