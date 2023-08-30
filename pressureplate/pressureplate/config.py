from ray.rllib.algorithms.ppo import PPOConfig

# config = PPOConfig()
# print(f'config.to_dict: {config.to_dict}')
# print(f'config.exploration_config: {config.exploration}')

config = PPOConfig()  
config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3)  
config = config.resources(num_gpus=0)  
config = config.rollouts(num_rollout_workers=4)  
print(config.to_dict())
