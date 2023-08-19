import sys
import numpy as np
from environment import PressurePlate


def sample_random_action(env):
    """Sample random action for agent."""
    return env.action_space.sample()

def sample_random_actions(env, obs):
    """Samples random actions for each agent."""

    # actions = {
    #     a_idx: sample_random_action(env)
    #     for a_idx in np.arange(env.n_agents)
    # }

    agents = np.arange(env.n_agents)
    actions = env.action_space.sample()

    agent_actions = {
        a_idx: action
        for a_idx, action in zip(agents, actions)
    }

    return agent_actions


# init values
kwargs= {
    'height': 15,
    'width': 9,
    'n_agents': 4,
    'sensor_range': 4,
    'layout': 'linear'
}

# customized layout
kwargs= {
    'height': 7,
    'width': 9,
    'n_agents': 2,
    'sensor_range': 4,
    'layout': 'customized'
}

env = PressurePlate(**kwargs)
obs = env.reset()
# # print(f"Original Env: \n {obs} \n")
env.render()
input()
sys.exit()
# actions = sample_random_actions(env, obs)
actions = {0: 3, 1: 3, 2: 2, 3: 2}
print(f"Actions: {actions} \n")
obs, rew, done, info = env.step(actions)
# # print(f"1 Step: \n {obs} \n")
env.render()
input()
obs, rew, done, info = env.step(actions)
# # print(f"2 Step: \n {obs} \n")
env.render()
input()
sys.exit()