import os
from constants import ROOT
from ray.rllib.algorithms.algorithm import Algorithm
from multi_agent_environment import MultiAgentPressurePlate
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
    env = MultiAgentPressurePlate(env_config)
    
    print(' Reset Env \n')
    obs, info = env.reset()
    env.render()
    input()

    print(' Simulating Policy \n')
    n_steps = 200
    for step in range(n_steps):
        print('##############')
        print(f'## STEP: {step} ##')
        print('##############')
        print()
        print('BEFORE ACTION')
        print(f'obs: {obs}')
        print(f'info: {info}')
        print()
        action_dict = {}
        for agent in obs.keys():
            action = algo.compute_single_action(
                obs[agent],
                # TODO generalize this using a policy_mapping_fn
                policy_id=f"agent_{agent}_policy",
                explore=False
            )
            action_dict[agent] = action
        print(f'ACTIONS: {action_dict}')
        print()
        print('AFTER ACTIONS')
        obs, reward, terminated, truncated, info = env.step(action_dict)
        print(f'rewards: { {agent: round(reward[agent], 5) for agent in reward.keys()} }')
        print(f'terminated: {terminated}')
        print(f'truncated: {truncated}')
        env.render()
        input()
        if terminated['__all__'] or truncated['__all__']:
            print('################')
            print('## TERMINATED ##')
            print('################')
            print()
            print(f'Total Rewards: { {agent: round(reward[agent], 5) for agent in reward.keys()} } \n')
            break
