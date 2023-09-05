from typing import Dict, List, Optional
from env_configs import ENV_CONFIGS

def print_training_result(
        result: Dict,
    ) -> None:
    # Dictionaries to print
    for m in ["policy_reward_min", "policy_reward_mean", "policy_reward_max"]:
        print(f'{m}:')
        for key, value in result[m].items():
            print(f"    {key}: {value}")
    # Single values to print
    for m in ["episode_len_mean"]:
        print(f'{m}: {result[m]}')

def get_env_config(
        env_name: str
    ) -> Dict:
    assert env_name in ENV_CONFIGS, f"There is no configuration named {env_name}. Check env_configs.py for supported configurations."
    return ENV_CONFIGS[env_name]

def check_entity(entity: str) -> Optional[ValueError]:
    if entity not in ['agents', 'walls', 'doors', 'plates', 'goals', 'escapes']:
        raise ValueError(f"""
            Invalid entity passed.
            Valid entities include 'agents', 'walls', 'doors', 'plates', 'goals', or 'escapes'.
            Got entity={entity}.
        """)