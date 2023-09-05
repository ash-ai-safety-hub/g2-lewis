
import numpy as np
from assets import LAYERS
from utils import check_entity
from entity import IPDAgent, MarketAgent, GridAgent, Entity
from typing import Dict
from gymnasium import spaces
import random

def get_obs_IPD(agents: [IPDAgent]) -> np.ndarray:
    """ 
    returns the last action taken by each player
        0 = Lie, 1 = Confess, -1 = Not yet taken an action
    """
    if (agents[0].y == 3):
        return np.array([-1,-1], dtype=np.float32)
    return np.array([agents[0].y, agents[1].y], dtype=np.float32)

def get_obs_IPD_noisy(agent: IPDAgent, agents: [IPDAgent], p: float) -> np.ndarray:
    """ 
    returns the last action taken by each player
        0 = Lie, 1 = Confess, -1 = Not yet taken an action
    """
    if (agents[0].y == 3):
        return np.array([-1,-1], dtype=np.float32)
    obs = np.array([agents[0].y, agents[1].y], dtype=np.float32)
    if random.uniform(0,1) < p:
        other_agent_id = (agent.id + 1) % 2
        obs[other_agent_id] = (obs[other_agent_id] + 1) % 2
    return obs

def get_obs_market(agents: [IPDAgent]) -> np.ndarray:
    """ 
    returns the last action taken by each player
        1-5 = The last price they set, 6 = Not yet taken an action
    """
    if (agents[0].y == 0): # if they have not yet played their first action
        return np.array([6, 6], dtype=np.float32)
    return np.array([agents[0].x + 1, agents[1].x + 1], dtype=np.float32)

def get_obs_sensor(agent: Entity, grid_size: (int, int), sensor_range: int, grid: np.ndarray) -> np.ndarray:
    """ 
    returns a flattened array of the bitmaps of each layer in the grid
    """
    # When the agent's vision, as defined by self.sensor_range,
    # goes off of the grid, we pad the grid-version of the observation.
    # Get padding.
    padding = _get_padding(agent, grid_size, sensor_range)
    # Add padding.
    _agents  = _pad_entity('agents' , padding, grid)
    _walls   = _pad_entity('walls'  , padding, grid)
    _doors   = _pad_entity('doors'  , padding, grid)
    _plates  = _pad_entity('plates' , padding, grid)
    _goals   = _pad_entity('goals'  , padding, grid)
    _escapes = _pad_entity('escapes', padding, grid)
    # Concatenate grids.
    obs = np.concatenate((_agents, _walls, _doors, _plates, _goals, _escapes, np.array([agent.x, agent.y])), axis=0, dtype=np.float32)
    # Flatten and return.
    obs = np.array(obs).reshape(-1)
    return obs




# Helper function for sensor based observation functions
def _get_padding(agent: Entity, grid_size: (int, int), sensor_range: int) -> Dict:
    x, y = agent.x, agent.y
    pad = sensor_range * 2 // 2
    padding = {}
    padding['x_left'] = max(0, x - pad)
    padding['x_right'] = min(grid_size[1] - 1, x + pad)
    padding['y_up'] = max(0, y - pad)
    padding['y_down'] = min(grid_size[0] - 1, y + pad)
    padding['x_left_padding'] = pad - (x - padding['x_left'])
    padding['x_right_padding'] = pad - (padding['x_right'] - x)
    padding['y_up_padding'] = pad - (y - padding['y_up'])
    padding['y_down_padding'] = pad - (padding['y_down'] - y)
    return padding

# Helper function for sensor based observation functions
def _pad_entity(entity: str, padding: Dict, grid: np.ndarray) -> np.ndarray:
    check_entity(entity)
    # For all objects but walls, we pad with zeros.
    # For walls, we pad with ones, as edges of the grid act in the same way as walls.
    padding_fn = np.zeros
    if entity == 'walls':
        padding_fn = np.ones
    # Get grid for entity.
    entity_grid = grid[LAYERS[entity], padding['y_up']:padding['y_down'] + 1, padding['x_left']:padding['x_right'] + 1]
    # Pad left.
    entity_grid = np.concatenate((padding_fn((entity_grid.shape[0], padding['x_left_padding'])), entity_grid), axis=1)
    # Pad right.
    entity_grid = np.concatenate((entity_grid, padding_fn((entity_grid.shape[0], padding['x_right_padding']))), axis=1)
    # Pad up.
    entity_grid = np.concatenate((padding_fn((padding['y_up_padding'], entity_grid.shape[1])), entity_grid), axis=0)
    # Pad down.
    entity_grid = np.concatenate((entity_grid, padding_fn((padding['y_down_padding'], entity_grid.shape[1]))), axis=0)
    # Flatten and return.
    entity_grid = entity_grid.reshape(-1)
    return entity_grid










def get_observation_space_sensor(agents: [GridAgent], sensor_range: int, grid_size: (int,int)):
    return spaces.Dict(
        {agent.id: spaces.Box(
            # All values will be 0.0 or 1.0 other than an agent's position.
            low=0.0,
            # An agent's position is constrained by the size of the grid.
            high=float(max([grid_size[0], grid_size[1]])),
            # An agent can see the {sensor_range} units in each direction (including diagonally) around them,
            # meaning they can see a square grid of {sensor_range} * 2 + 1 units.
            # They have a grid of this size for each of the 6 entities: agents, walls, doors, plates, goals, and escapes.
            # Plus they know their own position, parametrized by 2 values.
            shape=((sensor_range * 2 + 1) * (sensor_range * 2 + 1) * 6 + 2,),
            dtype=np.float32
        ) for agent in agents}
    )

def get_observation_space_IPD(agents: [IPDAgent]):
    return spaces.Dict(
        {agent.id: spaces.Box(
            # All values will be 0.0 for L or 1.0 for C or -1.0 for the start observation
            low=-1.0,
            high=1.0,
            # Each agent sees a tuple of the actions last round
            shape=(2,),
            dtype=np.float32
        ) for agent in agents}
    )

def get_observation_space_market(agents: [MarketAgent]):
    return spaces.Dict(
        {agent.id: spaces.Box(
            # Values will be the prices set last round by each agent 1-5 or 6 for start observation
            low=1.0,
            high=6.0,
            # Each agent sees a tuple of the actions last round
            shape=(2,),
            dtype=np.float32
        ) for agent in agents}
    )