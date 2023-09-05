import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from constants import NUM_TRAINING_ITERATIONS, CHECKPOINT_FREQUENCY, KL_END
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

def stop_fn(trial_id: str, result: dict) -> bool:
    # Function dictating when an experiment should end
    tooLong = result["training_iteration"] >= NUM_TRAINING_ITERATIONS
    converged0 = result['info']['learner']['agent_0_policy']['learner_stats']['kl'] <= KL_END
    converged1 = result['info']['learner']['agent_1_policy']['learner_stats']['kl'] <= KL_END
    return tooLong #or (result["training_iteration"] >= 15 and converged0 and converged1)

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
            lr=tune.grid_search([0.1, 0.05]) # Grid search through learning rates
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

    tuner = ray.tune.Tuner(
        "PPO",
        param_space=config,
        run_config=air.RunConfig(
            stop=stop_fn,
            checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True)#,checkpoint_frequency=CHECKPOINT_FREQUENCY),
        ),
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    best_checkpoint = best_result.checkpoint
    run, checkpoint = best_checkpoint.path.split('/')[-2:]
    print(f"Run demo of best checkpoint using: \n python demo.py --env_name {args.env_name} --run PPO/{run} --checkpoint {checkpoint} \n")