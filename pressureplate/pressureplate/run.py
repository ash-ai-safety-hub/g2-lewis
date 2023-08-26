import sys
import numpy as np
from environment import PressurePlate


# customized layout
kwargs = {
    'height': 7,
    'width': 9,
    'n_agents': 1,
    'sensor_range': 4,
    'layout': 'customized'
}

kwargs = {
    'height': 7,
    'width': 9,
    'n_agents': 1,
    'sensor_range': 1,
    'layout': "customized"
}

env = PressurePlate(**kwargs)
obs, info = env.reset()
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