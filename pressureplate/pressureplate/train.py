import ray
from ray.rllib.algorithms import ppo
from ray.tune.logger import pretty_print
from environment import PressurePlate
from ray.tune.registry import register_env


if __name__ == "__main__":

    print('\n Ray Init \n')
    ray.init()

    # def env_creator(env_config):
    #     print(f"\n Env Config: {env_config} \n")
    #     # height = env_config["height"]
    #     # width = env_config["width"]
    #     # n_agents = env_config["n_agents"]
    #     # sensor_range = env_config["sensor_range"]
    #     # layout = env_config["layout"]
    #     height = 7
    #     width = 9
    #     n_agents = 1
    #     sensor_range = 1
    #     layout = "customized"
    #     return PressurePlate(height, width, n_agents, sensor_range, layout)

    # print('\n Register Env \n')
    # register_env("PressurePlate", env_creator)

    print('\n Algo \n')
    # algo = ppo.PPO(env="PressurePlate")
    env_config = {
        'height': 7,
        'width': 9,
        'n_agents': 1,
        'sensor_range': 1,
        'layout': "customized"
    }
    algo = ppo.PPO(
        env=PressurePlate,
        config={
            "env_config": env_config,
        }
    )

    # print('\n Config \n')
    # config = (
    #     PPOConfig()
    #     .environment(
    #         env=PressurePlate,
    #         env_config=env_config
    #     )
    # )
    # print('\n Build \n')
    # algo = config.build()

    # def render_episode():
    #     env = SimpleCorridor({"corridor_length": 5})
    #     obs, info = env.reset()
    #     terminated = truncated = False
    #     total_reward = 0.0
    #     while not terminated and not truncated:
    #         # Render environment.
    #         env.render()
    #         # Compute a single action, given the current observation
    #         # from the environment.
    #         action = algo.compute_single_action(obs)
    #         # Apply the computed action in the environment.
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         # Sum up rewards for reporting purposes.
    #         total_reward += reward
    #     # Report results.
    #     print(f"Played 1 episode; total-reward={total_reward} \n")

    print('\n Train \n')
    for i in range(1):
        print(f'Training Iteration {i} \n')
        result = algo.train()
        print('Result \n')
        print(pretty_print(result))
        # render_episode()

    print('\n Stop \n')
    algo.stop()
