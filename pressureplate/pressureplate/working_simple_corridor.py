import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print


print('\n Define Class \n')
class SimpleCorridor(gym.Env):
    def __init__(self, config):
        self.end_pos = config["corridor_length"]
        self.cur_pos = 0
        self.action_space = gym.spaces.Discrete(2)  # right/left
        self.observation_space = gym.spaces.Discrete(self.end_pos)

    def reset(self, *, seed=None, options=None):
        self.cur_pos = 0
        return self.cur_pos, {}

    def step(self, action):
        if action == 0 and self.cur_pos > 0:  # move right (towards goal)
            self.cur_pos -= 1
        elif action == 1:  # move left (towards start)
            self.cur_pos += 1
        if self.cur_pos >= self.end_pos:
            return 0, 1.0, True, True, {}
        else:
            return self.cur_pos, -0.1, False, False, {}
    
    def render(self):
        corridor = ["_"] * self.end_pos
        corridor[self.end_pos - 1] = "G"
        corridor[self.cur_pos] = "x"
        corridor = [''.join(corridor)]
        print(corridor)


if __name__ == "__main__":

    print('\n Ray Init \n')
    ray.init()

    print('\n Config \n')
    config = (
        PPOConfig()
        .environment(
            env=SimpleCorridor,
            env_config={"corridor_length": 5}
        )
    )
    print('\n Build \n')
    algo = config.build()


    def render_episode():
        env = SimpleCorridor({"corridor_length": 5})
        obs, info = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        while not terminated and not truncated:
            # Render environment.
            env.render()
            # Compute a single action, given the current observation
            # from the environment.
            action = algo.compute_single_action(obs)
            # Apply the computed action in the environment.
            obs, reward, terminated, truncated, info = env.step(action)
            # Sum up rewards for reporting purposes.
            total_reward += reward
        # Report results.
        print(f"Played 1 episode; total-reward={total_reward} \n")

    print('\n Train \n')
    for i in range(3):
        print(f'Training Iteration {i} \n')
        result = algo.train()
        print('Result \n')
        print(pretty_print(result))
        render_episode()


    print('\n Stop \n')
    algo.stop()
