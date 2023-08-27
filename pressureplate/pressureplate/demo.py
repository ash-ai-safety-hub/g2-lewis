import os
from constants import ROOT
from ray.rllib.algorithms.algorithm import Algorithm
from environment import PressurePlate
import argparse
from utils import get_env_config

parser = argparse.ArgumentParser()
parser.add_argument(
     "--env_name", type=str, required=True, help="The PressurePlate configuration to use. See env_configs.py for supported configurations."
)
parser.add_argument(
    "--run", type=str, required=True, help=f"The folder in {ROOT} containing the checkpoint to use."
)
parser.add_argument(
    "--checkpoint", type=str, required=True, help="The checkpoint to use."
)

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"\n Running with following CLI options: {args} \n")

    checkpoint_path = os.path.join(ROOT, args.run, args.checkpoint)
    print(f'\n Pulling Policy From Checkpoint: {checkpoint_path} \n')
    
    print('\n Restoring Policy From Checkpoint \n')
    algo = Algorithm.from_checkpoint(checkpoint_path)

    print('\n Creating Env \n')
    env_config = get_env_config(args.env_name)
    env = PressurePlate(env_config)
    
    print('\n Reset Env \n')
    obs, info = env.reset()

    print('\n Simulating Policy \n')
    sum_reward = 0
    n_steps = 5
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
