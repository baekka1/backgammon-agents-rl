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

This agent is a TD($\lambda$) program that learns a neural value function and uses 2-ply search to chose moves. I decided to create an agent class called `BackgammonAgent`, which contains the model parameters for the `BackgammonValueNet` as well as all functions which are related to doing the training of the model. 

To train the model, I first initalized a `BackgammonAgent` agent. Then for each episode, I created a batch of games and for each game, select an action via 2-ply search, compute the reward, batch forward, then do the lambda update.

To run the training script: `python tdlambda.py`

# Agent 3
