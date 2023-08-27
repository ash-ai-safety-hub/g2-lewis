import ray
from ray.rllib.algorithms.ppo import PPOConfig
from environment import PressurePlate
from constants import NUM_TRAINING_ITERATIONS
from utils import print_training_result
import argparse
from utils import get_env_config

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env_name", type=str, required=True, help="The PressurePlate configuration to use. See env_configs.py for supported configurations."
)

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    print('\n Ray Init \n')
    ray.init()

    print('\n Setting env_config \n')
    env_config = get_env_config(args.env_name)

    print('\n Config \n')
    config = (
        PPOConfig()
        .environment(
            env=PressurePlate,
            env_config=env_config
        )
        .rollouts(
            num_rollout_workers=2,
            num_envs_per_worker=1
        )
        .resources(
            num_gpus=0,
            num_cpus_per_worker=2,
            num_gpus_per_worker=0
        )
    )

    print('\n Build Algo \n')
    algo = config.build()

    # print('\n Train \n')
    # for i in range(NUM_TRAINING_ITERATIONS):
    #     print(f'Training Iteration {i} \n')
    #     result = algo.train()
    #     print_training_result(result)

    #     if i % 2 == 0:
    #         checkpoint_dir = algo.save()
    #         print(f"Checkpoint saved in directory {checkpoint_dir}")
        
    #     print()

    print('\n Stop \n')
    algo.stop()
