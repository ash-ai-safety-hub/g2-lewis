import ray
from ray.rllib.algorithms.ppo import PPOConfig
from constants import NUM_TRAINING_ITERATIONS, CHECKPOINT_FREQUENCY, EXPLORE, LR, KL_END
from utils import print_training_result
import argparse
from utils import get_env_config
from ray.rllib.policy.policy import PolicySpec
from environment import MultiAgentPressurePlate
from constants import ROOT
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.algorithm import Algorithm
import matplotlib.pyplot as plt
from collusion_metrics import collusion_metric_IPD

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env_name", type=str, required=True, help="The PressurePlate configuration to use. See env_configs.py for supported configurations."
)

def simulate_policy_from_algo(algo, env_name: str, iteration_stopped_at):

    env_config = get_env_config(env_name)
    env = MultiAgentPressurePlate(env_config)
    obs, info = env.reset()
    
    rewards = [0, 0] # Tally up rewards throughout the game
    n_steps = 100
    for step in range(n_steps):
        action_dict = {}
        for agent in obs.keys():
            action = algo.compute_single_action(
                obs[agent],
                # TODO generalize this using a policy_mapping_fn
                policy_id=f"agent_{agent}_policy",
                explore=False
            )
            action_dict[agent] = action
        obs, reward, terminated, truncated, info = env.step(action_dict)
        rewards[0] = rewards[0] + reward[0]
        rewards[1] = rewards[1] + reward[1]

    # Display results about how this policy performed, how long it was trained for, and how it was trained
    print(f"[{EXPLORE}, {iteration_stopped_at}, {rewards}, {LR}, {collusion_metric_IPD(rewards, n_steps)}, {KL_END}]")

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
            lr=LR
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
            explore=EXPLORE
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

    # Keep track of the kl scores of each agent through the training process
    kl0 = []
    kl1 = []

    print('\n Train \n')
    i = 0
    converged = False
    while i < NUM_TRAINING_ITERATIONS and (not converged):
        print('############################')
        print(f'### Training Iteration {i + 1} ###')
        print('############################')
        result = algo.train()
        # algo.workers.foreach_worker(lambda worker: worker.get_policy().get_weights())
        print()
        # print(f'results: {results}')
        print_training_result(result)
        agent_0_kl = result['info']['learner']['agent_0_policy']['learner_stats']['kl']
        agent_1_kl = result['info']['learner']['agent_1_policy']['learner_stats']['kl']
        kl0.append(agent_0_kl)
        kl1.append(agent_1_kl)
        converged = agent_0_kl < KL_END and agent_1_kl < KL_END
        print()
        if (i + 1) % NUM_TRAINING_ITERATIONS == 0 or (i + 1) % CHECKPOINT_FREQUENCY == 0 or converged:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir} \n")
            run, checkpoint = checkpoint_dir.split('/')[-2:]
            print(f"See tensorboard of results using \n tensorboard --logdir={ROOT}/{run} \n")
            print(f"Run demo of checkpoint using: \n python demo.py --env_name {args.env_name} --run {run} --checkpoint {checkpoint} \n")
        i += 1
    print('Stop \n')

    simulate_policy_from_algo(algo, args.env_name, i)
    algo.stop()

    # Plot the kl scores of each agent over time
    plt.plot(kl0)
    plt.plot(kl1)
    plt.xlabel('Training Iteration')
    plt.ylabel('KL')
    plt.show()
