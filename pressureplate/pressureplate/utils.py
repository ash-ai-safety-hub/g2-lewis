from typing import Dict, List, Optional
from env_configs import ENV_CONFIGS

def print_training_result(
        result: Dict,
        metrics: List = ["episode_reward_min", "episode_reward_mean", "episode_reward_max", "episode_len_mean"]
    ) -> None:
    for m in metrics:
        print(f'{m}: {round(result[m], 2)}')

def get_env_config(
        env_name: str
    ) -> Dict:
    assert env_name in ENV_CONFIGS, f"There is no configuration named {env_name}. Check env_configs.py for supported configurations."
    return ENV_CONFIGS[env_name]

def check_entity(entity: str) -> Optional[ValueError]:
    if entity not in ['agents', 'walls', 'doors', 'plates', 'goals']:
        raise ValueError(f"""
            Invalid entity passed.
            Valid entities include 'agents', 'walls', 'doors', 'plates', or 'goals'.
            Got entity={entity}.
        """)