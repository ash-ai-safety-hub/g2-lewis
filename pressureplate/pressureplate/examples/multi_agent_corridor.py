import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

from working_simple_corridor import SimpleCorridor
from ray.rllib.env.multi_agent_env import make_multi_agent

from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.policy import PolicySpec

from ray import air, tune

print('\n Make Multi Agent Env \n')
ma_simple_corridor_cls = make_multi_agent( lambda config: SimpleCorridor(config) )
print('\n Instantiate Multi Agent Env \n')
ma_simple_corridor = ma_simple_corridor_cls( {"corridor_length": 5, "num_agents": 2} )
# obs = ma_simple_corridor.reset()
# print(obs)

print('\n Ray Init \n')
ray.init()

print('\n Config \n')
config = (
    PPOConfig()
    .environment(
        env=ma_simple_corridor_cls,
        env_config={"corridor_length": 5}
    )
    .multi_agent(
        policies={
            "learnable_policy": PolicySpec(
                config=PPOConfig.overrides(framework_str="tf")
            ),
            "random": PolicySpec(policy_class=RandomPolicy),
        },
        policy_mapping_fn=lambda agent_id, *args, **kwargs: [
            "learnable_policy",
            "random",
        ][agent_id % 2],
        # We wouldn't have to specify this here as the RandomPolicy does
        # not learn anyways (it has an empty `learn_on_batch` method), but
        # it's good practice to define this list here either way.
        policies_to_train=["learnable_policy"]
    )
)

print('\n Build \n')
algo = config.build()

print('\n Train \n')
for i in range(3):
    print(f'Training Iteration {i} \n')
    result = algo.train()
    print('Result \n')
    print(pretty_print(result))
    ma_simple_corridor.render()


# print('\n Stop \n')
# stop = {
#     "training_iteration": 5
# }

# print('\n Tuning \n')
# results = tune.Tuner(
#     "PPO",
#     param_space=config.to_dict(),
#     run_config=air.RunConfig(stop=stop, verbose=1),
# ).fit()

