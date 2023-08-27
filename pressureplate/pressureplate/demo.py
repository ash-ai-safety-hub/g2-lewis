import gymnasium as gym
import os
from constants import ROOT
from ray.rllib.algorithms.algorithm import Algorithm
from environment import PressurePlate


if __name__ == "__main__":

    run = 'PPO_PressurePlate_2023-08-27_11-11-40gzybjy0p'
    checkpoint = 'checkpoint_000003'
    checkpoint_path = os.path.join(ROOT, run, checkpoint)
    print(f'\n Pulling Policy From Checkpoint: {checkpoint_path} \n')
    
    print('\n Restoring Policy From Checkpoint \n')
    algo = Algorithm.from_checkpoint(checkpoint_path)

    print('\n Creating Env \n')
    # TODO use a registered env
    env_config = {
        'height': 7,
        'width': 9,
        'n_agents': 1,
        'sensor_range': 1,
        'layout': "customized"
    }
    env = PressurePlate(env_config=env_config)
    
    print('\n Reset Env \n')
    obs, info = env.reset()

    print('\n Simulating Policy \n')
    sum_reward = 0
    n_steps = 20
    for step in range(n_steps):
        print(f'step: {step}')
        action = algo.compute_single_action(obs)
        print(f'action: {action}')
        obs, reward, terminated, truncated, info = env.step(action)
        print(f'obs: {obs}')
        print(f'reward: {reward}')
        print(f'terminated: {terminated}')
        print(f'truncated: {truncated}')
        print(f'info: {info}')
        sum_reward += reward
        env.render()
        if terminated or truncated:
            print(f'sum_reward: {sum_reward}')
            obs, info = env.reset()
            sum_reward = 0
        input()
