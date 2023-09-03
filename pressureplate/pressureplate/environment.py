from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from actions import GridActions, IPDActions, MarketActions
from assets import LAYOUTS, LAYERS
from observations import get_obs_sensor, get_obs_IPD, get_obs_market
from observations import get_observation_space_IPD, get_observation_space_market, get_observation_space_sensor
from rewards import get_rewards_escape_and_split_treasure, get_rewards_IPD, get_rewards_market
from ray.rllib.env.env_context import EnvContext
import numpy as np
from utils import check_entity
from entity import Entity, GridAgent, IPDAgent, MarketAgent, Plate, Door, Wall, Goal, Escape    # used in _reset_entity
import sys
from constants import AGENT_TYPE_MARKET, AGENT_TYPE_GRID, AGENT_TYPE_IPD
from constants import OBSERVATION_METHOD_IPD, OBSERVATION_METHOD_MARKET, OBSERVATION_METHOD_SENSOR
from constants import REWARD_METHOD_ESCAPE_AND_SPLIT_TREASURE, REWARD_METHOD_IPD, REWARD_METHOD_MARKET


class MultiAgentPressurePlate(MultiAgentEnv):

    def __init__(self, env_config: EnvContext):
        super().__init__()

        self.layout = LAYOUTS[env_config['layout']]
        self.grid_size = (env_config['height'], env_config['width'])
        self.sensor_range = env_config['sensor_range']
        self.agent_type = env_config['agent_type']
        self.reward_method = env_config['reward_method']
        self.observation_method = env_config['observation_method']

        # Setup agents of the right class
        if self.agent_type == AGENT_TYPE_GRID:
            self.agent_class = GridAgent
            self.num_actions = len(GridActions)
        elif self.agent_type == AGENT_TYPE_IPD:
            self.agent_class = IPDAgent
            self.num_actions = len(IPDActions)
        elif self.agent_type == AGENT_TYPE_MARKET:
            self.agent_class = MarketAgent
            self.num_actions = len(MarketActions)
        self.agents = [self.agent_class(i, pos[0], pos[1]) for i, pos in enumerate(self.layout['AGENTS'])]
        self._agent_ids = [agent.id for agent in self.agents]

        # Setup action space such that it matches the agent type
        self.action_space = spaces.Dict(
            {agent.id: spaces.Discrete(self.num_actions) for agent in self.agents}
        )
        
        # Setup observation space such that it matches the observation type
        if self.observation_method == OBSERVATION_METHOD_SENSOR:
            self.observation_space = get_observation_space_sensor(self.agents, self.sensor_range, self.grid_size)
        elif self.observation_method == OBSERVATION_METHOD_IPD:
            self.observation_space = get_observation_space_IPD(self.agents)
        elif self.observation_method == OBSERVATION_METHOD_MARKET:
            self.observation_space = get_observation_space_market(self.agents)

        # TODO use the gamma in PPOConfig.training
        self.gamma = 0.98

        self._rendering_initialized = False
        self.viewer = None

        # Set initial conditions.
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # Wipe grid.
        self._wipe_grid()

        # Reset timesteps.
        # TODO use the timestep in algo
        self.timestep = 0

        # Put entities in their starting positions.
        self._reset_entity('agents')
        self._reset_entity('walls')
        self._reset_entity('doors')
        self._reset_entity('plates')
        self._reset_entity('goals')
        self._reset_entity('escapes')

        obs, info = {}, {}
        for agent in self.agents:
            obs[agent.id] = self._get_obs(agent)
            info[agent.id] = {}

        return obs, info

    def step(self, action_dict):

        # Take actions.
        for agent_id, action in action_dict.items():
            self.agents[agent_id].take_action(action, env=self)

        # Calculate reward.
        reward = {}
        for agent in self.agents:
            reward[agent.id] = self._get_reward(agent)

        # Update environment by (1) opening doors for plates that are pressed and (2) updating goals that have been achieved.
        if self.agent_type == AGENT_TYPE_GRID:
            self._update_plates_and_doors()
            self._update_goals()
            self._update_crushed_agents()

        # Get new observations for active agents.
        obs = {}
        for agent in self.agents:
            if (not self.agent_type == AGENT_TYPE_GRID) or (not agent.escaped and not agent.crushed):
                obs[agent.id] = self._get_obs(agent)

        # Check for game termination, which happens when all agents escape or time runs out.
        # TODO update, see here for motivation: https://github.com/ray-project/ray/blob/master/rllib/examples/env/multi_agent.py
        terminated, truncated = {}, {}
        for agent in self.agents:
            terminated[agent.id] = self.agent_type == AGENT_TYPE_GRID and (agent.escaped or agent.crushed)
            truncated[agent.id] = self.agent_type == AGENT_TYPE_GRID and (agent.escaped or agent.crushed)
        terminated["__all__"] = np.all([terminated[agent.id] for agent in self.agents])
        truncated["__all__"] = np.all([truncated[agent.id] for agent in self.agents])
        # TODO use tune instead of train to handle this, but for now...
        if self.timestep > 100:
            terminated["__all__"] = True
            truncated["__all__"] = True

        # Pass info.
        info = {}
        for agent in self.agents:
            if (not self.agent_type == AGENT_TYPE_GRID) or (not agent.escaped and not agent.crushed):
                info[agent.id] = {}

        # Increment timestep.
        self.timestep += 1

        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        if not self._rendering_initialized:
            self._init_render()
        return self.viewer.render(self, mode == 'rgb_array')

    def _wipe_grid(self):
        self.grid = np.zeros((len(LAYERS), *self.grid_size))

    def _reset_entity(self, entity: str) -> None:
        check_entity(entity)
        # Reset entity to empty list.
        setattr(self, entity, [])
        # Get class of entity. See entity.py for class definitions.
        if entity == "agents":
            entity_class = self.agent_class
        else:
            entity_class = getattr(sys.modules[__name__], entity[:-1].capitalize())    # taking away 's' at end of entity argument
        # Add values from assets.py to the grid.
        for id, pos in enumerate(self.layout[entity.upper()]):
            setattr(self, entity, getattr(self, entity) + [entity_class(id, pos[0], pos[1])])
            self.grid[LAYERS[entity], pos[1], pos[0]] = 1

    
    def _get_obs(self, agent: Entity) -> np.ndarray:
        if self.observation_method == OBSERVATION_METHOD_SENSOR:
            return get_obs_sensor(agent, self.grid_size, self.sensor_range, self.grid)
        elif self.observation_method == OBSERVATION_METHOD_IPD:
            return get_obs_IPD(self.agents)
        elif self.observation_method == OBSERVATION_METHOD_MARKET:
            return get_obs_market(self.agents)

    def _update_plates_and_doors(self) -> None:
        agents_pos = [[agent.x, agent.y] for agent in self.agents]
        for plate in self.plates:
            plate_pos = [plate.x, plate.y]
            if np.any([plate_pos == agent_pos for agent_pos in agents_pos]):
                plate.pressed = True
                self.doors[plate.id].open = True
                plate.ever_pressed = True
            else:
                plate.pressed = False
                self.doors[plate.id].open = False
    
    def _update_goals(self) -> None:
        agents_pos = [[agent.x, agent.y] for agent in self.agents]
        for goal in self.goals:
            if not goal.achieved:    # only have to check goals that haven't been achieved
                goal_pos = [goal.x, goal.y]
                if np.any([goal_pos == agent_pos for agent_pos in agents_pos]):
                    goal.achieved = True

    def _update_crushed_agents(self):
        for agent in self.agents:
            if not agent.escaped and not agent.crushed:
                agent_pos = [agent.x, agent.y]
                closed_doors_pos = [[door.x, door.y] for door in self.doors if not door.open]
                for closed_door_pos in closed_doors_pos:
                    if agent_pos == closed_door_pos:
                        agent.crushed = True
                        break
    
    def _get_reward(self, agent: Entity) -> float:
        if self.reward_method == REWARD_METHOD_IPD:
            return get_rewards_IPD(agent, self.agents)
        elif self.reward_method == REWARD_METHOD_ESCAPE_AND_SPLIT_TREASURE:
            return get_rewards_escape_and_split_treasure(agent, self.agents)
        elif self.reward_method == REWARD_METHOD_MARKET:
            return get_rewards_market(agent, self.agents)
    
    def _init_render(self):
        from rendering import Viewer
        self.viewer = Viewer(self.grid_size)
        self._rendering_initialized = True
