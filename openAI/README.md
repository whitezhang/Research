# openAI

### Background
The openAI gym is a good toolkit that provides environment for Reinforcement Learning problem. However, bugs exist in 2D box game(e.g. forzenLake). In order to use RL technique to play game, this project is created to make it.

### textGame

The textGame aims to give a basic tutorial about RL and now it supports Q-learning and SARSA.

###### envs
- This part contains the basic environment the game needs. You can create a new 2D box game by changing the **frozenLake.py** if you want.

###### q_s.py
```
python q_s.py [1|2|3] [q|s] [r|n]
```
while **q** means Q-learning, **s** is SARSA. You can add **r** if you want to check the results. Otherwise, use **n** instead.

