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
        'height': 7,
        'width': 9,
        'sensor_range': 5
    },

    "TwoAgent-IPD": {
        'layout': 'IPD',
        'agent_type': 'IPD',
        'height': 4,
        'width': 3,
        'sensor_range': 5
    }
    
}
