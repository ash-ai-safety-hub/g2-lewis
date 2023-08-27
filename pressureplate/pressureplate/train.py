import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from environment import PressurePlate
from ray.tune.registry import register_env
from constants import NUM_TRAINING_ITERATIONS
from utils import print_training_result


if __name__ == "__main__":

    print('\n Ray Init \n')
    ray.init()

    # TODO register env in __init__.py
    env_config = {
        'height': 7,
        'width': 9,
        'n_agents': 1,
        'sensor_range': 1,
        'layout': "customized"
    }

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

    print('\n Train \n')
    for i in range(NUM_TRAINING_ITERATIONS):
        print(f'Training Iteration {i} \n')
        result = algo.train()
        print_training_result(result)

        if i % 2 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")
        
        print()

    print('\n Stop \n')
    algo.stop()
