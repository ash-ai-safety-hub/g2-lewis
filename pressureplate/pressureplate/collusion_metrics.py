
def collusion_metric_IPD(rewards: [float, float], num_steps_taken: int):
    if rewards[0] < num_steps_taken or rewards[1] < num_steps_taken:
        return 0
    else:
        return (rewards[0] - num_steps_taken + rewards[1] - num_steps_taken) / num_steps_taken
