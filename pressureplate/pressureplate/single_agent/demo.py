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
    print(f' Pulling Policy From Checkpoint: {checkpoint_path} \n')
    
    print(' Restoring Policy From Checkpoint \n')
    algo = Algorithm.from_checkpoint(checkpoint_path)

    print('\n Creating Env \n')
    env_config = get_env_config(args.env_name)
    env = PressurePlate(env_config)
    
    print(' Reset Env \n')
    obs, info = env.reset()

    print(' Simulating Policy \n')
    sum_reward = 0
    n_steps = 30
    for step in range(n_steps):
        print('##############')
        print(f'## STEP: {step} ##')
        print('##############')
        print()
        print('BEFORE ACTION')
        print(f'obs: {obs}')
        print(f'info: {info}')
        print()
        action = algo.compute_single_action(obs)
        print(f'ACTION: {action}')
        print()
        print('AFTER ACTION')
        obs, reward, terminated, truncated, info = env.step(action)
        print(f'reward: {round(reward, 5)}')
        print(f'terminated: {terminated}')
        print(f'truncated: {truncated}')
        sum_reward += reward
        env.render()
        input()
        if terminated or truncated:
            print('## #############')
            print('## TERMINATED ##')
            print('## #############')
            print()
            print(f'Total Rewards: {round(sum_reward, 5)} \n')
            break
