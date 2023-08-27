from typing import Dict, List

def print_training_result(
        result: Dict,
        metrics: List = ["episode_reward_min", "episode_reward_mean", "episode_reward_max", "episode_len_mean"]
    ) -> None:
    for m in metrics:
        print(f'{m}: {round(result[m], 2)}')