## RL Simulations
Objective is to train a Reinforcement Learning model to solve the Cartpole gym environment with an excellent score.

Implemented Hill Climbing Agent to train a set of weights and learn of score high on the cartpole environment with continuous observation space and discrete action space.

### How to run?
1. **requirements.txt** - file contains the necessary python libraries to run the project
2. **main.py** - main file with the training loop and env & agent
3. **env.py** - OpenAI environment is initialised here
4. **agent.py** - hill climbing agent with its model and train implementations

```
$> pip install -r requirements.txt
```
This will install the necessary python libraries to run the project

```
$> python main.py
OR
$> python main.py <numEpisodes>
OR
$> python main.py <numEpisodes> <humanDisplay?>
```
This will run the project with **numEpisodes** number of episodes and displays the RL agents runs on the screen if **humanDisplay?** is set to **True**.
