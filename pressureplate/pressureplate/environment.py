import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from ray.rllib.env.env_context import EnvContext
import numpy as np
import sys
from actions import Actions
from assets import LAYERS, LAYOUTS

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
            # They have a grid of this size for each of the 4 entities: walls, doors, plates, goal.
            # Plus they know their own position, parametrized by 2 values.
            shape=((self.sensor_range * 2 + 1) * (self.sensor_range * 2 + 1) * 4 + 2,),
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
        self.agent_order = list(range(self.n_agents))
        self.viewer = None

        self.room_boundaries = np.unique(np.array(self.layout['WALLS'])[:, 1]).tolist()[::-1]
        self.room_boundaries.append(-1)

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)

        # Wipe grid
        self._wipe_grid()

        # Put entities in their starting positions
        self._reset_entity('agents')
        self._reset_entity('walls')
        self._reset_entity('doors')
        self._reset_entity('plates')
        self._reset_entity('goals')

        return self._get_obs(), {}

    def step(self, actions):

        # Randomize order of agents' actions
        np.random.shuffle(self.agent_order)

        # TODO fix this workaround that solves for actions being an int rather than a dict
        actions = {0: actions}

        for i in self.agent_order:
            proposed_pos = [self.agents[i].x, self.agents[i].y]

            if actions[i] == 0:
                proposed_pos[1] -= 1
                if not self._detect_collision(proposed_pos):
                    self.agents[i].y -= 1

            elif actions[i] == 1:
                proposed_pos[1] += 1
                if not self._detect_collision(proposed_pos):
                    self.agents[i].y += 1

            elif actions[i] == 2:
                proposed_pos[0] -= 1
                if not self._detect_collision(proposed_pos):
                    self.agents[i].x -= 1

            elif actions[i] == 3:
                proposed_pos[0] += 1
                if not self._detect_collision(proposed_pos):
                    self.agents[i].x += 1

            else:
                # NOOP
                pass

        for i, plate in enumerate(self.plates):
            if not plate.pressed:
                if [plate.x, plate.y] == [self.agents[plate.id].x, self.agents[plate.id].y]:
                    plate.pressed = True
                    self.doors[plate.id].open = True

            else:
                if [plate.x, plate.y] != [self.agents[plate.id].x, self.agents[plate.id].y]:
                    plate.pressed = False
                    self.doors[plate.id].open = False

        # Detecting goal completion
        r = []
        for agent in self.agents:
            r.append([agent.x, agent.y] == [self.goals[0].x, self.goals[0].y])
        got_goal = np.sum(r) > 0

        if got_goal:
            self.goals[0].achieved = True

        # TODO fix rewards function to return int dict rather than list
        rewards = self._get_rewards()
        reward = rewards[0]

        # return self._get_obs(), self._get_rewards(), [self.goals.achieved] * self.n_agents, [self.goals.achieved] * self.n_agents, {}
        return self._get_obs(), reward, self.goals[0].achieved, self.goals[0].achieved, {}

    def _reset_entity(self, entity: str) -> None:
        assert entity in ['agents', 'walls', 'doors', 'plates', 'goals'], \
            f"Expecting entity in ['agents', 'walls', 'doors', 'plates', 'goals']. Got entity={entity}."

        setattr(self, entity, [])

        # Get class of entity. See entity.py for class definitions.
        entity_class = getattr(sys.modules[__name__], entity[:-1].capitalize())    # taking away 's' at end of entity argument

        for i, ent in enumerate(self.layout[entity.upper()]):
            setattr(self, entity, getattr(self, entity) + [entity_class(i, ent[0], ent[1])])
            if entity == 'doors':
                # TODO make doors like the other entities
                for j in range(len(ent[0])):
                    self.grid[LAYERS[entity], ent[1][j], ent[0][j]] = 1
            else:
                self.grid[LAYERS[entity], ent[1], ent[0]] = 1


    def _get_obs(self):
        obs = []

        for agent in self.agents:
            x, y = agent.x, agent.y
            pad = self.sensor_range * 2 // 2

            x_left = max(0, x - pad)
            x_right = min(self.grid_size[1] - 1, x + pad)
            y_up = max(0, y - pad)
            y_down = min(self.grid_size[0] - 1, y + pad)

            x_left_padding = pad - (x - x_left)
            x_right_padding = pad - (x_right - x)
            y_up_padding = pad - (y - y_up)
            y_down_padding = pad - (y_down - y)

            # When the agent's vision, as defined by self.sensor_range, goes off of the grid, we
            # pad the grid-version of the observation. For all objects but walls, we pad with zeros.
            # For walls, we pad with ones, as edges of the grid act in the same way as walls.
            # For padding, we follow a simple pattern: pad left, pad right, pad up, pad down
            # Agents
            _agents = self.grid[LAYERS['agents'], y_up:y_down + 1, x_left:x_right + 1]

            _agents = np.concatenate((np.zeros((_agents.shape[0], x_left_padding)), _agents), axis=1)
            _agents = np.concatenate((_agents, np.zeros((_agents.shape[0], x_right_padding))), axis=1)
            _agents = np.concatenate((np.zeros((y_up_padding, _agents.shape[1])), _agents), axis=0)
            _agents = np.concatenate((_agents, np.zeros((y_down_padding, _agents.shape[1]))), axis=0)
            _agents = _agents.reshape(-1)

            # Walls
            _walls = self.grid[LAYERS['walls'], y_up:y_down + 1, x_left:x_right + 1]

            _walls = np.concatenate((np.ones((_walls.shape[0], x_left_padding)), _walls), axis=1)
            _walls = np.concatenate((_walls, np.ones((_walls.shape[0], x_right_padding))), axis=1)
            _walls = np.concatenate((np.ones((y_up_padding, _walls.shape[1])), _walls), axis=0)
            _walls = np.concatenate((_walls, np.ones((y_down_padding, _walls.shape[1]))), axis=0)
            _walls = _walls.reshape(-1)

            # Doors
            _doors = self.grid[LAYERS['doors'], y_up:y_down + 1, x_left:x_right + 1]

            _doors = np.concatenate((np.zeros((_doors.shape[0], x_left_padding)), _doors), axis=1)
            _doors = np.concatenate((_doors, np.zeros((_doors.shape[0], x_right_padding))), axis=1)
            _doors = np.concatenate((np.zeros((y_up_padding, _doors.shape[1])), _doors), axis=0)
            _doors = np.concatenate((_doors, np.zeros((y_down_padding, _doors.shape[1]))), axis=0)
            _doors = _doors.reshape(-1)

            # Plate
            _plates = self.grid[LAYERS['plates'], y_up:y_down + 1, x_left:x_right + 1]

            _plates = np.concatenate((np.zeros((_plates.shape[0], x_left_padding)), _plates), axis=1)
            _plates = np.concatenate((_plates, np.zeros((_plates.shape[0], x_right_padding))), axis=1)
            _plates = np.concatenate((np.zeros((y_up_padding, _plates.shape[1])), _plates), axis=0)
            _plates = np.concatenate((_plates, np.zeros((y_down_padding, _plates.shape[1]))), axis=0)
            _plates = _plates.reshape(-1)

            # Goal
            _goal = self.grid[LAYERS['goals'], y_up:y_down + 1, x_left:x_right + 1]

            _goal = np.concatenate((np.zeros((_goal.shape[0], x_left_padding)), _goal), axis=1)
            _goal = np.concatenate((_goal, np.zeros((_goal.shape[0], x_right_padding))), axis=1)
            _goal = np.concatenate((np.zeros((y_up_padding, _goal.shape[1])), _goal), axis=0)
            _goal = np.concatenate((_goal, np.zeros((y_down_padding, _goal.shape[1]))), axis=0)
            _goal = _goal.reshape(-1)

            # Concat
            obs.append(np.concatenate((_agents, _plates, _doors, _goal, np.array([x, y])), axis=0, dtype=np.float32))

        obs = np.array(obs).reshape(-1)
        return obs

    def _detect_collision(self, proposed_position):
        """Need to check for collision with (1) grid edge, (2) walls, (3) closed doors (4) other agents"""
        # Grid edge
        if np.any([
            proposed_position[0] < 0,
            proposed_position[1] < 0,
            proposed_position[0] >= self.grid_size[1],
            proposed_position[1] >= self.grid_size[0]
        ]):
            return True

        # Walls
        for wall in self.walls:
            if proposed_position == [wall.x, wall.y]:
                return True

        # Closed Door
        for door in self.doors:
            if not door.open:
                for j in range(len(door.x)):
                    if proposed_position == [door.x[j], door.y[j]]:
                        return True

        # Other agents
        for agent in self.agents:
            if proposed_position == [agent.x, agent.y]:
                return True

        return False

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

    def _get_rewards(self):
        rewards = []

        for agent in self.agents:

            if [agent.x, agent.y] == [self.goals[0].x, self.goals[0].y]:
                reward = 1
            else:
                reward = 0
            
            rewards.append(reward)

        return rewards

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
