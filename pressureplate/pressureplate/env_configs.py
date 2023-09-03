from constants import AGENT_TYPE_GRID, AGENT_TYPE_IPD, AGENT_TYPE_MARKET
from constants import OBSERVATION_METHOD_SENSOR, OBSERVATION_METHOD_IPD, OBSERVATION_METHOD_MARKET
from constants import REWARD_METHOD_ESCAPE_AND_SPLIT_TREASURE, REWARD_METHOD_IPD, REWARD_METHOD_MARKET

"""
Each config specifies a particular environment and the agents actions, rewards and observations.

- layout:               All layouts can be found in 'assets.py',
- agent_type:           'grid', 'IPD', 'market'
- observation_method:   'sensor', 'IPD', 'market'
- sensor_range:         int specifying view distance of agents using 'sensor' observation_method
- reward_method:        'EscapeAndSplitTreasure', 'IPD', 'market'
- height:               int specifying the height of the grid world,
- width:                int specifying the width of the grid world,
"""

ENV_CONFIGS = {

    "SingleAgent-v0": { # Not complete
        'layout': 'BasicOneAgent',
        'height': 7,
        'width': 9,
        'sensor_range': 1
    },

    "TwoAgent-v0": { # Not complete
        'layout': 'BasicTwoAgent',
        'height': 7,
        'width': 9,
        'sensor_range': 0
    },

    "TwoAgent-v1": {
        'layout': 'CooperativeTwoAgent',
        'agent_type': AGENT_TYPE_GRID,
        'observation_method': OBSERVATION_METHOD_SENSOR,
        'sensor_range': 5,
        'reward_method': REWARD_METHOD_ESCAPE_AND_SPLIT_TREASURE,
        'height': 7,
        'width': 9,
    },

    "TwoAgent-IPD": {
        'layout': 'IPD',
        'agent_type': AGENT_TYPE_IPD,
        'observation_method': OBSERVATION_METHOD_IPD,
        'sensor_range': 5,
        'reward_method': REWARD_METHOD_IPD,
        'height': 4,
        'width': 3,
    },

    "TwoAgent-Market": {
        'layout': 'Market',
        'agent_type': AGENT_TYPE_MARKET,
        'observation_method': OBSERVATION_METHOD_MARKET,
        'sensor_range': 5,
        'reward_method': REWARD_METHOD_MARKET,
        'height': 4,
        'width': 5,
    }
    
}
