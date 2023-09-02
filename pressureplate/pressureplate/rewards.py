import numpy as np
from entity import GridAgent, IPDAgent

def get_rewards_escape_and_split_treasure(agent: GridAgent, agents: [GridAgent]):
    # Agents who escape evenly split the total treasure they found.
    if not agent.escaped:
        return 0
    total_treasue = np.sum([a.treasure for a in agents if a.escaped])
    n_escaped_agents = np.sum([a.escaped for a in agents])
    return total_treasue / n_escaped_agents

def get_rewards_IPD(agent: IPDAgent, agents: [IPDAgent]):
    if agents[0].y == agents[1].y and agents[0].y == 0: # LL
        return 3
    elif agents[0].y == agents[1].y and agents[0].y == 1: # CC
        return 1
    else:
        if agent.y == 0: # This agent did L, and the other did C
            return 0
        else: # This agent did C, and the other did L
            return 4

def get_rewards_market(agent: IPDAgent, agents: [IPDAgent]):
    reward_profile = [0, 0]
    if (agents[0].x > agents[1].x):
        reward_profile = [0, agents[1].x + 1]
    elif (agents[0].x < agents[1].x):
        reward_profile = [agents[0].x + 1, 0]
    else:
        reward_profile = [(agents[0].x + 1)/2.0, (agents[1].x + 1)/2.0]
    return reward_profile[agent.id]