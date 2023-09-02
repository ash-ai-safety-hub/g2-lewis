
import numpy as np
from assets import LAYOUTS, LAYERS
from utils import check_entity
from entity import IPDAgent, Entity, Plate, Door, Wall, Goal, Escape
from typing import Dict, Tuple

def get_obs_IPD(agents: [IPDAgent]) -> np.ndarray:
    if (agents[0].y == 3): # if they have not yet played their first action
        return np.array([-1,-1], dtype=np.float32)
    return np.array([agents[0].y, agents[1].y], dtype=np.float32)

def get_obs_market(agents: [IPDAgent]) -> np.ndarray:
    if (agents[0].y == 0): # if they have not yet played their first action
        return np.array([6, 6], dtype=np.float32)
    return np.array([agents[0].x + 1, agents[1].x + 1], dtype=np.float32)

def get_obs_sensor(agent: Entity, grid_size: (int, int), sensor_range: int, grid: np.ndarray) -> np.ndarray:
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