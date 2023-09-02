ENV_CONFIGS = {

    "SingleAgent-v0": {
        'layout': 'BasicOneAgent',
        'height': 7,
        'width': 9,
        'sensor_range': 1
    },

    "TwoAgent-v0": {
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
    }
    
}
