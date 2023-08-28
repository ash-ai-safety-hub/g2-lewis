import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from ray.rllib.env.env_context import EnvContext
import numpy as np
import sys
from actions import Actions
from entity import Agent, Plate, Door, Wall, Goal    # used in _reset_entity
from assets import LAYERS, LAYOUTS
from utils import check_entity
from typing import Dict

# TODO generalize to more than 1 goal


class PressurePlate(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, env_config: EnvContext):

        self.grid_size = (env_config['height'], env_config['width'])
        self.n_agents = env_config['n_agents']
        self.sensor_range = env_config['sensor_range']

        self.action_space = Discrete(len(Actions))

        self.observation_space = Box(
            low=0.0,
            # All values will be 0.0 or 1.0 other than an agent's position.
            # An agent's position is constrained by the size of the grid.
            high=float(max([self.grid_size[0], self.grid_size[1]])),
            # An agent can see the {sensor_range} units in each direction (including diagonally) around them,
            # meaning they can see a square grid of {sensor_range} * 2 + 1 units.
            # They have a grid of this size for each of the 5 entities: agents, walls, doors, plates, goals.
            # Plus they know their own position, parametrized by 2 values.
            shape=((self.sensor_range * 2 + 1) * (self.sensor_range * 2 + 1) * 5 + 2,),
            dtype=np.float32
        )

        self.agents = []
        self.plates = []
        self.walls = []
        self.doors = []
        self.goals = []

        self._rendering_initialized = False

        self._wipe_grid()
        self.layout = LAYOUTS[env_config['layout']]

        self.max_dist = np.linalg.norm(np.array([0, 0]) - np.array([2, 8]), 1)
        # self.agent_order = list(range(self.n_agents))
        self.viewer = None

        self.room_boundaries = np.unique(np.array(self.layout['WALLS'])[:, 1]).tolist()[::-1]
        self.room_boundaries.append(-1)

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)

        # Wipe grid.
        self._wipe_grid()

        # Put entities in their starting positions.
        self._reset_entity('agents')
        self._reset_entity('walls')
        self._reset_entity('doors')
        self._reset_entity('plates')
        self._reset_entity('goals')

        return self._get_obs(), {}

    def step(self, actions):

        # Randomize order of agents' actions.
        # np.random.shuffle(self.agent_order)

        # TODO fix this workaround that solves for actions being an int rather than a dict
        actions = {0: actions}

        # Take actions.
        for agent in self.agents:
            action = actions[agent.id]
            agent.take_action(action, env=self)

        # Update environment by (1) opening doors for plates that are pressed and (2) updating goals that have been achieved.
        self._update_plates_and_doors()
        self._update_goals()

        # Get new observations.
        obs = self._get_obs()

        # Check for goal completion.
        if np.all([goal.achieved for goal in self.goals]):
            terminated = True
        else:
            terminated = False

        # TODO update this for the multi-agent case
        reward = 0
        for agent in self.agents:
            reward += self._get_reward(agent)

        # Pass info.
        info = {}

        return obs, reward, terminated, terminated, info

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

    def _update_plates_and_doors(self) -> None:
        agents_pos = [[agent.x, agent.y] for agent in self.agents]
        for plate in self.plates:
            plate_pos = [plate.x, plate.y]
            if np.any([plate_pos == agent_pos for agent_pos in agents_pos]):
                plate.pressed = True
                self.doors[plate.id].open = True
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

    def _get_obs(self):
        obs = []

        for agent in self.agents:

            # When the agent's vision, as defined by self.sensor_range, goes off of the grid, we
            # pad the grid-version of the observation. For all objects but walls, we pad with zeros.
            # For walls, we pad with ones, as edges of the grid act in the same way as walls.

            padding = self._get_padding(agent)

            _agents = self._pad_entity('agents', padding)
            _walls  = self._pad_entity('walls' , padding)
            _doors  = self._pad_entity('doors' , padding)
            _plates = self._pad_entity('plates', padding)
            _goals  = self._pad_entity('goals' , padding)

            obs.append(np.concatenate((_agents, _walls, _doors, _plates, _goals, np.array([agent.x, agent.y])), axis=0, dtype=np.float32))

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

    def _pad_entity(
        self,
        entity: str,
        padding: Dict
    ) -> np.ndarray:
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

    def _get_flat_grid(self):
        grid = np.zeros(self.grid_size)

        # Plate
        for plate in self.plates:
            grid[plate.y, plate.x] = 2

        # Walls
        for wall in self.walls:
            grid[wall.y, wall.x] = 3

        # Doors
        for door in self.doors:
            if door.open:
                grid[door.y, door.x] = 0
            else:
                grid[door.y, door.x] = 4

        # Goal
        grid[self.goals[0].y, self.goals[0].x] = 5

        # Agents
        for agent in self.agents:
            grid[agent.y, agent.x] = 1

        return grid

    def _get_reward(self, agent):
        goals_pos = [[goal.x, goal.y] for goal in self.goals]
        agent_pos = [agent.x, agent.y]
        if np.any([agent_pos == goal_pos for goal_pos in goals_pos]):
            reward = 1
        else:
            reward = 0
        return reward

    def _get_curr_room_reward(self, agent_y):
        for i, room_level in enumerate(self.room_boundaries):
            if agent_y > room_level:
                curr_room = i
                break
        return curr_room
    
    def _wipe_grid(self):
        self.grid = np.zeros((len(LAYERS), *self.grid_size))

    def _init_render(self):
        from rendering import Viewer
        self.viewer = Viewer(self.grid_size)
        self._rendering_initialized = True

    def render(self, mode='human'):
        if not self._rendering_initialized:
            self._init_render()
        return self.viewer.render(self, mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
