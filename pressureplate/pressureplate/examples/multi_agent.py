import gymnasium as gym
from typing import Tuple, Optional
from ray.rllib.utils.typing import MultiAgentDict

print("\n Importing Stuff \n")
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from working_simple_corridor import SimpleCorridor

print("\n Defining Class \n")
class MultiAgentCorridor(MultiAgentEnv):
    """Env of N independent agents, each of which exits after 25 steps."""

    metadata = {
        "render.modes": ["rgb_array"],
    }
    render_mode = "rgb_array"

    print("\n Init \n")
    def __init__(self):
        super().__init__()
        self.n_agents = 2
        self.agents = [SimpleCorridor({"corridor_length": 3}) for _ in range(self.n_agents)]
        self._agent_ids = set(range(self.n_agents))
        self.terminateds = set()
        self.truncateds = set()
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
        self.resetted = False

    print("\n Reset \n")
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """Resets the env and returns observations from ready agents.

        Args:
            seed: An optional seed to use for the new episode.

        Returns:
            New observations for each ready agent.
        """
        super().reset(seed=seed)

        self.resetted = True
        self.terminateds = set()
        self.truncateds = set()
        reset_results = [agent.reset() for agent in self.agents]
        return (
            {i: oi[0] for i, oi in enumerate(reset_results)},
            {i: oi[1] for i, oi in enumerate(reset_results)},
        )

    print("\n Step \n")
    def step(
        self,
        action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        """Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns:
            Tuple containing 1) new observations for
            each ready agent, 2) reward values for each ready agent. If
            the episode is just started, the value will be None.
            3) Terminated values for each ready agent. The special key
            "__all__" (required) is used to indicate env termination.
            4) Truncated values for each ready agent.
            5) Info values for each agent id (may be empty dicts).
        """
        obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}
        for i, action in action_dict.items():
            obs[i], rew[i], terminated[i], truncated[i], info[i] = self.agents[i].step(
                action
            )
            if terminated[i]:
                self.terminateds.add(i)
            if truncated[i]:
                self.truncateds.add(i)
        terminated["__all__"] = len(self.terminateds) == len(self.agents)
        truncated["__all__"] = len(self.truncateds) == len(self.agents)
        return obs, rew, terminated, truncated, info

    print("\n Render \n")
    def render(self) -> None:
        """Tries to render the environment."""
        for a in self.agents:
            print(f"Rendering Env for Agent {a}")
            a.render()
        return


print("\n Importing Packages \n")
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.examples.policy.random_policy import RandomPolicy

print("\n Writing env_creator Func \n")
def env_creator(env_config):
    return MultiAgentCorridor()

print("\n Creating Env\n ")
env = env_creator({})
print("\n Registering Env \n")
register_env("MutliAgentCorridor", env_creator)

print("\n Setting Config \n")
config = (
    PPOConfig()
    .environment("MutliAgentCorridor")
    .rollouts(num_rollout_workers=2)
    .multi_agent(
        # The multiagent Policy map.
        policies={
            # The Policy we are actually learning.
            "learnable_policy": PolicySpec(
                config=PPOConfig.overrides(framework_str="tf")
            ),
            # Random policy we are playing against.
            "random": PolicySpec(policy_class=RandomPolicy),
        },
        # Map to either random behavior or PR learning behavior based on
        # the agent's ID.
        policy_mapping_fn=lambda agent_id, *args, **kwargs: [
            "learnable_policy",
            "random",
        ][agent_id % 2],
        # We wouldn't have to specify this here as the RandomPolicy does
        # not learn anyways (it has an empty `learn_on_batch` method), but
        # it's good practice to define this list here either way.
        policies_to_train=["learnable_policy"],
    )
)

print("\n Building Algo \n")
algo = config.build()

print("\n Training Algo \n")
timesteps = 3
for i in range(timesteps):
    print("\n Training \n")
    results = algo.train()
    print(f"\n Iter: {i}; avg. reward={results['episode_reward_mean']} \n")


env = MultiAgentCorridor()
# Get the initial observation (should be: [0.0] for the starting position).
obs, info = env.reset()
terminated = truncated = False
total_reward = 0.0
# Play one episode.
print("OG Env")
env.render()
while not terminated and not truncated:
    # Compute a single action, given the current observation
    # from the environment.
    action = algo.compute_single_action(obs)
    print(action)
    # Apply the computed action in the environment.
    obs, reward, terminated, truncated, info = env.step(action)
    # Sum up rewards for reporting purposes.
    total_reward += reward
    # Render environment.
    env.render()
# Report results.
print(f"Played 1 episode; total-reward={total_reward}")