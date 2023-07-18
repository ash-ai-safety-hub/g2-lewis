from einops import reduce
import numpy as np
import nashpy as nash

# The purpose of this code is to analyze how the Nash equilibria of a game change when agents are allowed one step of memory.
# It takes a baseline two-player non-repeated general-sum Markov game as input and finds the corresponding two-round general-sum Markov game with a one round of memory.
# The new game represents different policies where agent actions can be conditioned on their one step of memory.
# Finding Nash Equilibria takes a very long time, can be hours if original matrix is 3x3 or larger.

class OnePeriodMemoryGame:
    def __init__(self, game):
        """
        Initialize the OnePeriodMemoryGame class.

        Parameters:
        - game: The baseline non-repeated general-sum Markov game. Two 2D np.arrays corresponding to the payoffs for the two agents (perhaps to be extended to more than 2 agents in future).
        """
        self.game = game
        self.action_dims = game.shape[1:]
        
        print("Finding Original Nash Equilibria")
        self.original_NE = get_nash_equilibria(self.game)
        self.original_deterministic_NE = reduce_to_deterministic(self.original_NE)
        self.original_NE_utilities = get_utilities(self.original_NE, self.game)

        print("Building New Policies")
        self.new_policies_0, self.new_policies_1 = next_level_policies(self.action_dims)
        print("Constructing One Step Game")
        self.one_step_game = self.game_with_one_period_memory()
        self.new_game_dims = self.one_step_game.shape[1:]
        self.actions_by_policy = self.resultant_actions_by_policy()
        print("Shape of new game: ", self.new_game_dims)
        
        print("Finding New NE")
        self.new_NE = get_nash_equilibria(self.one_step_game)
        self.new_deterministic_NE = reduce_to_deterministic(self.new_NE)
        print("Building New NE Actions")
        self.new_NE_actions = self.get_one_step_NE_actions()
        self.unique_NE_actions = np.unique(self.new_NE_actions, axis=0)
        print("Finding new NE utilities")
        self.final_NE_utilities = get_utilities(self.new_NE, self.one_step_game)


    def game_with_one_period_memory(self):
        """
        Construct a new 2-round game based on the baseline game, but allowing 1-step memory.

        This method constructs a new 2-round game where agents have a 1-step memory, allowing their actions to be conditioned on their memory. The first round action is based on no information, and the second round action can depend on their opponent's play in the first round.

        Returns:
        - The constructed 1-step memory game.
        """
        rewards = np.zeros((2, len(self.new_policies_0), len(self.new_policies_1)))
        for i, policy_0 in enumerate(self.new_policies_0):
            for j, policy_1 in enumerate(self.new_policies_1):
                    payoffs = self.find_utility_of_two_step_policy(policy_0, policy_1)
                    rewards[:, i, j] = payoffs
        return rewards


    def find_utility_of_two_step_policy(self, policy_0, policy_1):
        """
        Finds the payoff accumulated over the two rounds for each player, given their policies and the payoff matrix.

        Parameters:
        - policy_0: The two step policy of player 0, as described in next_level_policies description.
        - policy_1: Analogous input for player 1.

        Returns:
        - total_payoff: A tuple of length 2, containing the total 2-round payoff for each of player 0 and player 1.        
        """
        
        #  First decode the actions taken by the two players.
        actions = find_actions_given_policies(policy_0, policy_1)

        #  Now calculate the payoff of each player, given their actions.
        round_0_payoffs = self.game[:, actions[0][0], actions[0][1]]
        round_1_payoffs = self.game[:, actions[1][0], actions[1][1]]
        return round_0_payoffs + round_1_payoffs
    
    def get_one_step_NE_actions(self):
        """
        NOT FINISHED
        This does not work if actions are non-deterministic. Need to work on this.
        For now, I'm ignoring any NE that are non-deterministic.
        """
        actions = np.zeros((len(self.new_deterministic_NE), 2, 2))
        for i, NE in enumerate(self.new_deterministic_NE):
            encoded_policy_0 = self.new_policies_0[np.argmax(NE[0])]
            encoded_policy_1 = self.new_policies_1[np.argmax(NE[1])]
            actions[i, :] = find_actions_given_policies(encoded_policy_0, encoded_policy_1)
        return actions.astype(int)

    def resultant_actions_by_policy(self):
        """
        Returns the actions that would be played in the two-round version of the game for each combination of two-round policies.
        """
        actions = np.zeros((self.new_game_dims[0], self.new_game_dims[1], 2, 2))
        for policy_0_ind in range(self.new_game_dims[0]):
            policy_0 = single_1_array(policy_0_ind, self.new_game_dims[0])
            for policy_1_ind in range(self.new_game_dims[1]):
                policy_1 = single_1_array(policy_1_ind, self.new_game_dims[1])
                actions[policy_0_ind, policy_1_ind, :, :] = find_actions_given_policies(policy_0, policy_1)
        return actions


    def get_one_step_NE_actions_full(self):
        """
        Given the set of Nash Equilibria for the one step memory game, calculate the resultant two-round strategy in terms of the original actions. 
        self.actions_by_policy contains the actions resulting from policies with index policy_0_ind and policy_1_ind at self.actions_by_policy[policy_0_ind, policy_1_ind]. 
        Hence, we need only work out the probability of each of them taking place (with outer product) then dot the action vectors with these probabilities. 


        UNFINISHED - THE IDEA ABOVE DOESN'T WORK, since for each NE, a non-deterministic equilibrium for 2 players must be 3D
        """
        actions = np.zeros((self.new_NE.shape[0], 2, 2))
        for i, NE in enumerate(self.new_NE):
            policy_combo_probs = np.outer(NE[0], NE[1])
#            actions[i, :, :] = reduce(
#                policy_combo_probs * game_obj.actions_by_policy,
#                'i j a b -> a b',
#                reduction='sum'
#                )
            actions[i, :, :] = np.einsum('i j, i j a b -> a b',
                                         policy_combo_probs,
                                         game_obj.actions_by_policy)


    def get_utilities(self, deterministic_NE_array):
        """
        Again only doing this for the deterministic NE.
        """
        utilities = np.zeros((len(deterministic_NE_array), 2))
        for i, action in enumerate(deterministic_NE_array):
            first_round_payoffs = self.game[:, action[0][0], action[0][1]]
            second_round_payoffs = self.game[:, action[1][0], action[1][1]]
            utilities[i, :] = first_round_payoffs + second_round_payoffs
        return utilities

    def __str__(self):
        """
        Return a string representation of the OnePeriodMemoryGame object.
        """
        # Original game
        original_game_str = "Original Game:\n" + str(self.game) + "\n"

        # Original Nash equilibria
        original_NE_str = "Original Nash Equilibria:\n" + str(self.original_NE.round(2)) + "\n"

        # New one-step memory game
        one_step_game_str = "One-Step Memory Game:\n" + str(self.one_step_game) + "\n"

        # New Nash equilibria
        new_NE_str = "New Nash Equilibria:\n" + str(self.new_NE.round(2)) + "\n"

        # New Deterministic Nash equilibria
        new_det_NE_str = "New (Deterministic Only) Nash Equilibria:\n" + str(self.new_deterministic_NE.round(2)) + "\n"

        # New Nash equilibria actions
        new_NE_actions_str = "New (unique) Deterministic Nash Equilibria Actions:\n" + str(self.unique_NE_actions) + "\n"

        # New Nash Utilities
        new_NE_utilities_str = "The New NE achieve per-round utilities as follows:\n" + str(self.final_NE_utilities/2) + "\n"

        # Original Nash Utilities
        original_NE_utilities_str = "Compare these with original NE utilities:\n" + str(self.original_NE_utilities) + "\n"

        # Not using the deterministic Nash equilibria here. To do so, swap new_NE_str for new_det_NE_str
        return original_game_str + original_NE_str + one_step_game_str + new_NE_str + new_det_NE_str + new_NE_actions_str + new_NE_utilities_str + original_NE_utilities_str
    
    def __repr__(self):
        return str(self)


def get_utilities(policies, game):
    """
    Finds the utilities of a list of policies for player 0 and player 1, in a given game.
    """
    utilities = np.zeros((len(policies), 2))
    for i, action in enumerate(policies):
        selection_probs = np.outer(action[0], action[1])
        utilities[i, :] = np.sum(game * selection_probs, axis=(1,2))
    return utilities


def find_actions_given_policies(policy_0, policy_1):
    """
    Determines the actions of the two players, given their two-step policy encodings.
    
    Returns:
    Defining a_ri as the action of agent i in round r, this returns a np.array with
    [[a_00, a_01],
        [a_10, a_11]]
    """
    actions = np.zeros((2, 2))

    # First actions encoded simply by the first entry of each policy.
    player_0_first_action = policy_0[0]
    player_1_first_action = policy_1[0]
    
    actions[0][0] = player_0_first_action
    actions[0][1] = player_1_first_action

    # Second actions are conditional on the first action of opponent, encoded as described in next_level_policies function.
    actions[1][0] = policy_0[player_1_first_action + 1]
    actions[1][1] = policy_1[player_0_first_action + 1]
    
    return actions.astype(int)

def next_level_policies(action_space_dims):
    """
    Returns a list of possible strategies for both players, when they have one period of memory. 
    Each strategy for player i is of the following form: (a, b_0, b_1, ..., b_{k-1}), where k is the number of possible actions of player j, a is the initial action of player i, and b_l is the second-round action of player i given that player j played l in the first round.
        
    Parameters:
    - action_space_dims: The dimension of the base game. That is, a np.array with the number of possible actions for each of the two players.
    
    Returns:
    - policies_1: A list of lists, where each entry corresponds to a 2-round policy for player 0 as described above.
    - policies_2: Analogous result for player 1.
    """

    policies_0 = generate_lists_of_combos(action_space_dims[0], action_space_dims[1]+1)
    policies_1 = generate_lists_of_combos(action_space_dims[1], action_space_dims[0]+1)
    return policies_0, policies_1

def single_1_array(i, size):
    """
    Generates an array of zeros of size=size, with a 1 at index i.
    """
    arr = np.zeros(size)
    arr[i] = 1
    return arr.astype(int)


def generate_lists_of_combos(n, k):
    """
    Generate all lists of length k that contain the digits from 0 to n-1.

    Parameters:
    - n: The number of digits to consider.
    - k: The length of the lists to generate.

    Returns:
    - A list of all possible lists of length k containing digits from 0 to n-1.
    """
    if k == 0:
        return [[]]

    result = []
    for i in range(n):
        sublists = generate_lists_of_combos(n, k - 1)
        for sublist in sublists:
            result.append([i] + sublist)

    return result

def get_nash_equilibria(game, deterministic_only=False):
    """
    Find the Nash Equilibria of the input game.
    If deterministic_only, returns only deterministic NE.
    """
    nash_game = nash.Game(game[0], game[1])
    if not deterministic_only:
        # All equilibria
        nash_equilibria = list(nash_game.support_enumeration())
    else:
        # Only deterministic
        nash_equilibria = list(nash_game.vertex_enumeration())
    nash_equilibria = np.array(nash_equilibria)
    return nash_equilibria

def reduce_to_deterministic(policies):
    """
    Takes a list of policies, and returns only those that are deterministic (for all players).
    """
    deterministic_policies = policies[
        np.logical_and(
            reduce(policies[:, 0, :], 'b c -> b', reduction='max') == 1,
            reduce(policies[:, 1, :], 'b c -> b', reduction='max') == 1
        )
    ]
    return deterministic_policies


prisoners_dilemma = np.array([[[2, 0],
                               [3, 1]],
                               
                               [[2, 3],
                                [0, 1]]])

PD_flee = np.array([[[2, 0, 0],
                     [0, 2, 0],
                     [0, 3, 1]],

                     [[2, 0, 0],
                      [0, 2, 3],
                      [0, 0, 1]]])

turf_war = np.array([[[1, 5, 1],
                      [5, 1, 1],
                      [6, 6, 2]],

                      [[1, 5, 6],
                       [5, 1, 6],
                       [1, 1, 2]]])

chicken = np.array([[[0, 7],
                     [2, 6]],

                     [[0, 2],
                      [7, 6]]])


game_obj = OnePeriodMemoryGame(prisoners_dilemma)
game_obj