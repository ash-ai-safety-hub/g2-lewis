from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from actions import Actions
from assets import LAYOUTS, LAYERS
from ray.rllib.env.env_context import EnvContext
import numpy as np
from utils import check_entity
from entity import Agent, Plate, Door, Wall, Goal    # used in _reset_entity
from typing import Dict, Tuple
import sys


class MultiAgentPressurePlate(MultiAgentEnv):

    def __init__(self, env_config: EnvContext):
        super().__init__()

        self.layout = LAYOUTS[env_config['layout']]
        self.grid_size = (env_config['height'], env_config['width'])
        self.sensor_range = env_config['sensor_range']

        self.agents = [Agent(i, pos[0], pos[1]) for i, pos in enumerate(self.layout['AGENTS'])]
        self._agent_ids = [agent.id for agent in self.agents]

        self.action_space = spaces.Dict(
            {agent.id: spaces.Discrete(len(Actions)) for agent in self.agents}
        )
        self.observation_space = spaces.Dict(
            {agent.id: spaces.Box(
                low=0.0,
                high=float(max([self.grid_size[0], self.grid_size[1]])),
                shape=((self.sensor_range * 2 + 1) * (self.sensor_range * 2 + 1) * 5 + 2,),
                dtype=np.float32
            ) for agent in self.agents}
        )

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
        self.timestep = 0

        # Put entities in their starting positions.
        self._reset_entity('agents')
        self._reset_entity('walls')
        self._reset_entity('doors')
        self._reset_entity('plates')
        self._reset_entity('goals')

        obs, info = {}, {}
        for agent in self.agents:
            obs[agent.id] = self._get_obs(agent)
            info[agent.id] = {}

        return obs, info

    def step(self, action_dict):

        # Take actions.
        for agent in self.agents:
            action = action_dict[agent.id]
            agent.take_action(action, env=self)

        # Calculate reward.
        # TODO update so that all agents don't share the same reward.
        reward = {}
        for agent in self.agents:
            reward[agent.id] = self._get_reward(agent)

        # Update environment by (1) opening doors for plates that are pressed and (2) updating goals that have been achieved.
        self._update_plates_and_doors()
        self._update_goals()

        # Get new observations.
        obs = {}
        for agent in self.agents:
            obs[agent.id] = self._get_obs(agent)

        # Check for goal completion.
        # TODO update, see here for motivation: https://github.com/ray-project/ray/blob/master/rllib/examples/env/multi_agent.py
        done = self._check_goal_completion()
        terminated, truncated = {}, {}
        for agent in self.agents:
            terminated[agent.id] = done
            truncated[agent.id] = done
        terminated["__all__"] = np.all([terminated[agent] == True for agent in terminated.keys()])
        truncated["__all__"] = np.all([truncated[agent] == True for agent in truncated.keys()])

        # Pass info.
        info = {}
        for agent in self.agents:
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
        entity_class = getattr(sys.modules[__name__], entity[:-1].capitalize())    # taking away 's' at end of entity argument
        # Add values from assets.py to the grid.
        for id, pos in enumerate(self.layout[entity.upper()]):
            setattr(self, entity, getattr(self, entity) + [entity_class(id, pos[0], pos[1])])
            if entity == 'doors':
                # TODO make doors like the other entities
                for j in range(len(pos[0])):
                    self.grid[LAYERS[entity], pos[1][j], pos[0][j]] = 1
            else:
                self.grid[LAYERS[entity], pos[1], pos[0]] = 1

    def _get_obs(self, agent: Agent):
        # When the agent's vision, as defined by self.sensor_range, goes off of the grid, we
        # pad the grid-version of the observation. For all objects but walls, we pad with zeros.
        # For walls, we pad with ones, as edges of the grid act in the same way as walls.
        # Get padding.
        padding = self._get_padding(agent)
        # Add padding.
        _agents = self._pad_entity('agents', padding)
        _walls  = self._pad_entity('walls' , padding)
        _doors  = self._pad_entity('doors' , padding)
        _plates = self._pad_entity('plates', padding)
        _goals  = self._pad_entity('goals' , padding)
        # Concatenate grids.
        obs = np.concatenate((_agents, _walls, _doors, _plates, _goals, np.array([agent.x, agent.y])), axis=0, dtype=np.float32)
        # Flatten and return.
        obs = np.array(obs).reshape(-1)
        return obs
    
    def _get_padding(self, agent: Agent) -> Dict:
        x, y = agent.x, agent.y
        pad = self.sensor_range * 2 // 2
        padding = {}
        padding['x_left'] = max(0, x - pad)
        padding['x_right'] = min(self.grid_size[1] - 1, x + pad)
        padding['y_up'] = max(0, y - pad)
        padding['y_down'] = min(self.grid_size[0] - 1, y + pad)
        padding['x_left_padding'] = pad - (x - padding['x_left'])
        padding['x_right_padding'] = pad - (padding['x_right'] - x)
        padding['y_up_padding'] = pad - (y - padding['y_up'])
        padding['y_down_padding'] = pad - (padding['y_down'] - y)
        return padding

    def _pad_entity(self, entity: str, padding: Dict) -> np.ndarray:
        check_entity(entity)
        entity_grid = self.grid[LAYERS[entity], padding['y_up']:padding['y_down'] + 1, padding['x_left']:padding['x_right'] + 1]
        # Pad left.
        entity_grid = np.concatenate((np.zeros((entity_grid.shape[0], padding['x_left_padding'])), entity_grid), axis=1)
        # Pad right.
        entity_grid = np.concatenate((entity_grid, np.zeros((entity_grid.shape[0], padding['x_right_padding']))), axis=1)
        # Pad up.
        entity_grid = np.concatenate((np.zeros((padding['y_up_padding'], entity_grid.shape[1])), entity_grid), axis=0)
        # Pad down.
        entity_grid = np.concatenate((entity_grid, np.zeros((padding['y_down_padding'], entity_grid.shape[1]))), axis=0)
        # Flatten and return.
        entity_grid = entity_grid.reshape(-1)
        return entity_grid
    
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

    # TODO update this
    # Right now the game ends when all of the goals are acheived.
    def _check_goal_completion(self) -> bool:
        if np.all([goal.achieved for goal in self.goals]):
            done = True
        else:
            done = False
        return done
    
    def _get_reward(self, agent: Agent):
        agents_pos = [[agent.x, agent.y] for agent in self.agents]
        goals_pos = [[goal.x, goal.y] for goal in self.goals if not goal.achieved]
        plates_pos = [[plate.x, plate.y] for plate in self.plates if not plate.ever_pressed]
        reward = 0
        for agent_pos in agents_pos:
            for goal_pos in goals_pos:
                if agent_pos == goal_pos:
                    reward += 100 * self.gamma**self.timestep
            for plate_pos in plates_pos:
                if agent_pos == plate_pos:
                    reward += 1 * self.gamma**self.timestep
        return reward
    
    def _init_render(self):
        from rendering import Viewer
        self.viewer = Viewer(self.grid_size)
        self._rendering_initialized = True
