import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from ray.rllib.env.env_context import EnvContext
import numpy as np
from actions import Actions
from entity import Agent, Plate, Door, Wall, Goal
from assets import LAYERS, LINEAR, CUSTOMIZED


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
            # meaning they can see a square grid of {sensor_range} + 1 units.
            # They have a grid of this size for each of the 4 entities: walls, doors, plates, goal.
            # Plus they know their own position, parametrized by 2 values.
            # shape=(6,),
            shape=((self.sensor_range * 2 + 1) * (self.sensor_range * 2 + 1) * 4 + 2,),
            dtype=np.float32
        )

        self.agents = []
        self.plates = []
        self.walls = []
        self.doors = []
        self.goal = None

        self._rendering_initialized = False

        self._wipe_grip()
        if env_config['layout'] == 'linear':
            if self.n_agents == 4:
                self.layout = LINEAR['FOUR_PLAYERS']

            elif self.n_agents == 5:
                self.layout = LINEAR['FIVE_PLAYERS']

            elif self.n_agents == 6:
                self.layout = LINEAR['SIX_PLAYERS']
            else:
                raise ValueError(f'Number of agents given ({self.n_agents}) is not supported.')
            
        elif env_config['layout'] == 'customized':
            if self.n_agents == 1:
                self.layout = CUSTOMIZED['ONE_PLAYER']
            if self.n_agents == 2:
                self.layout = CUSTOMIZED['BASIC_TWO_PLAYER']

        self.max_dist = np.linalg.norm(np.array([0, 0]) - np.array([2, 8]), 1)
        self.agent_order = list(range(self.n_agents))
        self.viewer = None

        self.room_boundaries = np.unique(np.array(self.layout['WALLS'])[:, 1]).tolist()[::-1]
        self.room_boundaries.append(-1)
        print(f'self.room_boundaries: {self.room_boundaries}')

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)

        # Wipe grid
        self._wipe_grip()

        # Agents
        self.agents = []
        for i in range(self.n_agents):
            self.agents.append(Agent(i,
                                    self.layout['AGENTS'][self.agent_order[i]][0],
                                    self.layout['AGENTS'][self.agent_order[i]][1]))
            self.grid[LAYERS['agents'],
                    self.layout['AGENTS'][self.agent_order[i]][1],
                    self.layout['AGENTS'][self.agent_order[i]][0]] = 1

        # Walls
        self.walls = []
        for i, wall in enumerate(self.layout['WALLS']):
            self.walls.append(Wall(i, wall[0], wall[1]))
            self.grid[LAYERS['walls'], wall[1], wall[0]] = 1

        # Doors
        self.doors = []
        for i, door in enumerate(self.layout['DOORS']):
            self.doors.append(Door(i, door[0], door[1]))
            for j in range(len(door[0])):
                self.grid[LAYERS['doors'], door[1][j], door[0][j]] = 1

        # Plate
        self.plates = []
        for i, plate in enumerate(self.layout['PLATES']):
            self.plates.append(Plate(i, plate[0], plate[1]))
            self.grid[LAYERS['plates'], plate[1], plate[0]] = 1

        # Goal
        self.goal = []
        self.goal = Goal('goal', self.layout['GOAL'][0][0], self.layout['GOAL'][0][1])
        self.grid[LAYERS['goal'], self.layout['GOAL'][0][1], self.layout['GOAL'][0][0]] = 1

        return self._get_obs(), {}

    def step(self, actions):

        # Randomize order of agents' actions
        np.random.shuffle(self.agent_order)

        # TODO fix this workaround that solves for actions being an int rather than a dict
        actions = {0: actions}
        print(f'actions: {actions}')

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
            r.append([agent.x, agent.y] == [self.goal.x, self.goal.y])
        got_goal = np.sum(r) > 0

        if got_goal:
            self.goal.achieved = True

        # TODO fix rewards function to return int dict rather than list
        rewards = self._get_rewards()
        reward = rewards[0]

        # return self._get_obs(), self._get_rewards(), [self.goal.achieved] * self.n_agents, [self.goal.achieved] * self.n_agents, {}
        return self._get_obs(), reward, self.goal.achieved, self.goal.achieved, {}

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
            _goal = self.grid[LAYERS['goal'], y_up:y_down + 1, x_left:x_right + 1]

            _goal = np.concatenate((np.zeros((_goal.shape[0], x_left_padding)), _goal), axis=1)
            _goal = np.concatenate((_goal, np.zeros((_goal.shape[0], x_right_padding))), axis=1)
            _goal = np.concatenate((np.zeros((y_up_padding, _goal.shape[1])), _goal), axis=0)
            _goal = np.concatenate((_goal, np.zeros((y_down_padding, _goal.shape[1]))), axis=0)
            _goal = _goal.reshape(-1)

            # Concat
            print(f"Agents: {_agents}")
            print(f"Plates: {_plates}")
            print(f"Doors: {_doors}")
            print(f"Goal: {_goal}")
            obs.append(np.concatenate((_agents, _plates, _doors, _goal, np.array([x, y])), axis=0, dtype=np.float32))

        # return tuple(obs)
        print(f'type(obs): {type(obs)}, obs: {obs}')
        obs = np.array(obs).reshape(-1)
        print(f'type(obs): {type(obs)}, obs: {obs}')
        print(f'len(obs): {len(obs)}')
        return obs

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
        grid[self.goal.y, self.goal.x] = 5

        # Agents
        for agent in self.agents:
            grid[agent.y, agent.x] = 1

        return grid

    def _get_rewards(self):
        rewards = []

        # The last agent's desired location is the goal instead of a plate, so we use an if/else block
        # to break between the two cases
        for i, agent in enumerate(self.agents):

            if i == len(self.agents) - 1:
                plate_loc = self.goal.x, self.goal.y
            else:
                plate_loc = self.plates[i].x, self.plates[i].y
            print(f'plate_loc: {plate_loc}')

            curr_room = self._get_curr_room_reward(agent.y)
            print(f'curr_room: {curr_room}')

            agent_loc = agent.x, agent.y
            print(f'agent_loc: {agent_loc}')

            print(f'i: {i}')
            if i == curr_room:
                reward = - np.linalg.norm((np.array(plate_loc) - np.array(agent_loc)), 1) / self.max_dist
            else:
                reward = -len(self.room_boundaries)+1 + curr_room
            
            rewards.append(reward)
        return rewards

    def _get_curr_room_reward(self, agent_y):
        for i, room_level in enumerate(self.room_boundaries):
            if agent_y > room_level:
                curr_room = i
                break

        return curr_room
    
    # TODO see init and reset
    def _wipe_grip(self):
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
