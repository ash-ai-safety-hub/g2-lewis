from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from actions import GridActions, IPDActions
from assets import LAYOUTS, LAYERS
from observations import get_obs_sensor
from rewards import get_rewards_escape_and_split_treasure, get_rewards_IPD
from ray.rllib.env.env_context import EnvContext
import numpy as np
from utils import check_entity
from entity import Entity, GridAgent, IPDAgent, Plate, Door, Wall, Goal, Escape    # used in _reset_entity
from typing import Dict, Tuple
import sys


class MultiAgentPressurePlate(MultiAgentEnv):

    def __init__(self, env_config: EnvContext):
        super().__init__()

        self.layout = LAYOUTS[env_config['layout']]
        self.grid_size = (env_config['height'], env_config['width'])
        self.sensor_range = env_config['sensor_range']
        self.agent_type = env_config['agent_type']
        self.reward_method = env_config['reward_method']

        # Setup agents of the right type
        if self.agent_type == 'grid':
            self.agents = [GridAgent(i, pos[0], pos[1]) for i, pos in enumerate(self.layout['AGENTS'])]
        elif self.agent_type == 'IPD':
            self.agents = [IPDAgent(i, pos[0], pos[1]) for i, pos in enumerate(self.layout['AGENTS'])]
        
        self._agent_ids = [agent.id for agent in self.agents]

        # Setup action space such that it matches the agent type
        if self.agent_type == 'grid':
            self.action_space = spaces.Dict(
                {agent.id: spaces.Discrete(len(GridActions)) for agent in self.agents}
            )
        if self.agent_type == 'IPD':
            self.action_space = spaces.Dict(
                {agent.id: spaces.Discrete(len(IPDActions)) for agent in self.agents}
            )
        self.observation_space = spaces.Dict(
            {agent.id: spaces.Box(
                # All values will be 0.0 or 1.0 other than an agent's position.
                low=0.0,
                # An agent's position is constrained by the size of the grid.
                high=float(max([self.grid_size[0], self.grid_size[1]])),
                # An agent can see the {sensor_range} units in each direction (including diagonally) around them,
                # meaning they can see a square grid of {sensor_range} * 2 + 1 units.
                # They have a grid of this size for each of the 6 entities: agents, walls, doors, plates, goals, and escapes.
                # Plus they know their own position, parametrized by 2 values.
                shape=((self.sensor_range * 2 + 1) * (self.sensor_range * 2 + 1) * 6 + 2,),
                dtype=np.float32
            ) for agent in self.agents}
        )

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

        # TODO rearrange these so that agents can't move into places where gates will close

        # Take actions.
        for agent_id, action in action_dict.items():
            self.agents[agent_id].take_action(action, env=self)

        # Calculate reward.
        reward = {}
        for agent in self.agents:
            reward[agent.id] = self._get_reward(agent)

        # Update environment by (1) opening doors for plates that are pressed and (2) updating goals that have been achieved.
        self._update_plates_and_doors()
        self._update_goals()

        # Get new observations for active agents.
        obs = {}
        for agent in self.agents:
            if self.agent_type == 'IPD' or (not agent.escaped):
                obs[agent.id] = self._get_obs(agent)

        # Check for game termination, which happens when all agents escape or time runs out.
        # TODO update, see here for motivation: https://github.com/ray-project/ray/blob/master/rllib/examples/env/multi_agent.py
        terminated, truncated = {}, {}
        for agent in self.agents:
            terminated[agent.id] = self.agent_type == 'grid' and agent.escaped
            truncated[agent.id] = self.agent_type == 'grid' and agent.escaped
        terminated["__all__"] = np.all([terminated[agent.id] for agent in self.agents])
        truncated["__all__"] = np.all([truncated[agent.id] for agent in self.agents])
        # TODO use tune instead of train to handle this, but for now...
        if self.timestep > 100:
            terminated["__all__"] = True
            truncated["__all__"] = True

        # Pass info.
        info = {}
        for agent in self.agents:
            if self.agent_type == 'IPD' or (not agent.escaped):
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
        if entity != "agents":
            entity_class = getattr(sys.modules[__name__], entity[:-1].capitalize())    # taking away 's' at end of entity argument
        # Add values from assets.py to the grid.
        for id, pos in enumerate(self.layout[entity.upper()]):
            if entity == "agents":
                if self.agent_type == "Grid":
                    setattr(self, entity, getattr(self, entity) + [GridAgent(id, pos[0], pos[1])])
                if self.agent_type == "IPD":
                    setattr(self, entity, getattr(self, entity) + [IPDAgent(id, pos[0], pos[1])])
            else:
                setattr(self, entity, getattr(self, entity) + [entity_class(id, pos[0], pos[1])])
            if entity == 'doors':
                # TODO make doors like the other entities
                for j in range(len(pos[0])):
                    self.grid[LAYERS[entity], pos[1][j], pos[0][j]] = 1
            else:
                self.grid[LAYERS[entity], pos[1], pos[0]] = 1
    
    def _get_obs(self, agent: Entity) -> np.ndarray:
        return get_obs_sensor(agent, self.grid_size, self.sensor_range, self.grid)

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
    
    def _get_reward(self, agent: Entity):
        if self.reward_method == "IPD":
            return get_rewards_IPD(agent, self.agents)
        else:
            return get_rewards_escape_and_split_treasure(agent, self.agents)
    
    def _init_render(self):
        from rendering import Viewer
        self.viewer = Viewer(self.grid_size)
        self._rendering_initialized = True
