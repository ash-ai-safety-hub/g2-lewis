
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
        'agent_type': 'grid',
        'observation_method': 'sensor',
        'sensor_range': 5,
        'reward_method': 'EscapeAndSplitTreasure',
        'height': 7,
        'width': 9,
    },

    "TwoAgent-IPD": {
        'layout': 'IPD',
        'agent_type': 'IPD',
        'observation_method': 'IPD',
        'sensor_range': 5,
        'reward_method': 'IPD',
        'height': 4,
        'width': 3,
    },

    "TwoAgent-Market": {
        'layout': 'Market',
        'agent_type': 'market',
        'observation_method': 'market',
        'sensor_range': 5,
        'reward_method': 'market',
        'height': 4,
        'width': 5,
    }
    
}
