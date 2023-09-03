import ray
from ray.rllib.algorithms.ppo import PPOConfig
from constants import NUM_TRAINING_ITERATIONS, CHECKPOINT_FREQUENCY
from utils import print_training_result
import argparse
from utils import get_env_config
from ray.rllib.policy.policy import PolicySpec
from environment import MultiAgentPressurePlate
from constants import ROOT

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env_name", type=str, required=True, help="The PressurePlate configuration to use. See env_configs.py for supported configurations."
)

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"\n Running with following CLI options: {args}")

    print('\n Ray Init \n')
    ray.init()

    print('\n Setting env_config \n')
    env_config = get_env_config(args.env_name)

    print(' Config \n')
    config = (
        PPOConfig()
        .training(
            lr=0.001
        )
        .environment(
            env=MultiAgentPressurePlate,
            env_config=env_config
        )
        .framework(
            "tf"
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
        .exploration(
            explore=True
        )
        .multi_agent(
            policies={
                "agent_0_policy": PolicySpec(),
                "agent_1_policy": PolicySpec()
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: f"agent_{agent_id}_policy"
        )
    )

    print('\n Build Algo \n')
    algo = config.build()

    print('\n Train \n')
    for i in range(NUM_TRAINING_ITERATIONS):
        print('############################')
        print(f'### Training Iteration {i + 1} ###')
        print('############################')
        result = algo.train()
        # algo.workers.foreach_worker(lambda worker: worker.get_policy().get_weights())
        print()
        # print(f'results: {results}')
        print_training_result(result)
        print()
        if (i + 1) % NUM_TRAINING_ITERATIONS == 0 or (i + 1) % CHECKPOINT_FREQUENCY:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir} \n")
            run, checkpoint = checkpoint_dir.split('/')[-2:]
            print(f"See tensorboard of results using \n tensorboard --logdir={ROOT}/{run} \n")
            print(f"Run demo of checkpoint using: \n python demo.py --env_name {args.env_name} --run {run} --checkpoint {checkpoint} \n")

    print('Stop \n')
    algo.stop()
