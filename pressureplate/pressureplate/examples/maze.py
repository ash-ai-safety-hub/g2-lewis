import gymnasium as gym

from gymnasium.spaces import Discrete
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print


class Maze(gym.Env):

    seeker, goal = (0, 0), (4, 4)
    info = {'seeker': seeker, 'goal': goal}

    def __init__(self,  *args, **kwargs):
        self.action_space = Discrete(4)
        self.observation_space = Discrete(5*5)

    def step(self, action):
        """Take a step in a direction and return all available information."""
        if action == 0:  # move down
            self.seeker = (min(self.seeker[0] + 1, 4), self.seeker[1])
        elif action == 1:  # move left
            self.seeker = (self.seeker[0], max(self.seeker[1] - 1, 0))
        elif action == 2:  # move up
            self.seeker = (max(self.seeker[0] - 1, 0), self.seeker[1])
        elif action == 3:  # move right
            self.seeker = (self.seeker[0], min(self.seeker[1] + 1, 4))
        else:
            raise ValueError("Invalid action")

        return self.get_observation(), self.get_reward(), self.is_done(), self.is_done(), self.info

    def reset(self, *, seed=42, options=None):
        """Reset seeker and goal positions, return observations."""
        super().reset(seed=seed)
        self.seeker = (0, 0)
        self.goal = (4, 4)

        return self.get_observation(), {}

    def render(self):
        """Render the environment, e.g. by printing its representation."""
        grid = [['| ' for _ in range(5)] + ["|\n"] for _ in range(5)]
        grid[self.goal[0]][self.goal[1]] = '|G'
        grid[self.seeker[0]][self.seeker[1]] = '|S'
        print(''.join([''.join(grid_row) for grid_row in grid]))

    def get_observation(self):
        """Encode the seeker position as integer"""
        return 5 * self.seeker[0] + self.seeker[1]

    def get_reward(self):
        """Reward finding the goal"""
        return 1 if self.seeker == self.goal else 0

    def is_done(self):
        """We're done if we found the goal"""
        return self.seeker == self.goal
    

register_env(
    "maze", lambda _: Maze()
)

config = (
    PPOConfig()
    .environment(
        env="maze",
        render_env=True
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

algo = config.build()

for i in range(2):
    result = algo.train()
print('\n Result \n')
print(pretty_print(result))


env = Maze()
obs, info = env.reset()
terminated = truncated = False
total_reward = 0
while not terminated and not truncated:
    action = algo.compute_single_action(obs)
    print(f'Action: {action} \n')
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    print(f'Reward: {reward} \n')
    total_reward += reward


