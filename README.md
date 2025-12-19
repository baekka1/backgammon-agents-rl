# Reinforcement Learning Final Project
Authors: Katie Baek and Josh Gorniak

# Set Up
First, set up a virtual environment: `python3 -m venv .venv`  
Then, activate the environment: `source .venv/bin/activate`  
Download of the necessary libraries: `pip install -r requirements.txt`

# Agent 1
**Josh Gorniak**

# Agent 2
**Katie Baek**

This agent is a TD($\lambda$) program that learns a neural value function and uses 2-ply search to chose moves. I decided to create an agent class called `BackgammonAgent`, which contains the model parameters for the `BackgammonValueNet` as well as all functions which are related to doing the training of the model. The environment state is as described in the project, with 15 feature planes and 6 auxillery features. 

To train the model, I first initalized a `BackgammonAgent` agent. Then for each episode, I created a batch of games and for each game, select an action via 2-ply search, applies the moves in the engine, computes rewards and bootstrap values, and updates the value network using semi-gradient TD($\lambda$) with eligibility traces.  

Running `python tdlambda.py` trains an agent on 5 episodes with 5 games, and then does one real game with the trained agent.

# Agent 3

