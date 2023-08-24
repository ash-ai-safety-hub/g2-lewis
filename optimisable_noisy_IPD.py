import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def generate_P_matrix(p):
    """
    p: probability of error in memory of opponent's previous action
    returns: 4x16 tensor P such that P_{i,j} = Pr()
    """
    P_vector = torch.tensor([(1-p)**2, (1-p)*p, p*(1-p), p**2],
                            requires_grad=True).float()
    I = torch.eye(4)
    P_matrix = torch.kron(I, P_vector)#.reshape((4,4,4))
    return P_matrix

def generate_P_matrix_new(p):
    p1_matrix = torch.tensor([
        [1-p,p,0,0],
        [p,1-p,0,0],
        [0,0,1-p,p],
        [0,0,p,1-p]
    ], dtype=float).unsqueeze(-1)
    p2_matrix = torch.tensor([
        [1-p,0,p,0],
        [0,1-p,0,p],
        [p,0,1-p,0],
        [0,p,0,1-p]
    ], dtype=float).unsqueeze(-1)
    return torch.matmul(p1_matrix,p2_matrix.mT)





class NoisyIPD:
    def __init__(self, game, p_i, p_j, p, gamma):
        self.game = game.float()
        self.p_i = torch.nn.Parameter(p_i.float(), requires_grad=True)
        self.p_j = p_j.float()
        self.p = p
        self.gamma = gamma
        self.P = generate_P_matrix(self.p)
        self.Q = self.generate_Q_matrix()
        self.values_0 = self.find_values(0)
        self.values_1 = self.find_values(1)


    def generate_Q_submatrix(self, i_indices, j_indices):
        p_i_sub = self.p_i[i_indices]
        p_j_sub = self.p_j[j_indices]

        v1 = (1 - p_i_sub).outer(1 - p_j_sub).view(-1)
        v2 = (1 - p_i_sub).outer(p_j_sub).view(-1)
        v3 = p_i_sub.outer(1 - p_j_sub).view(-1)
        v4 = p_i_sub.outer(p_j_sub).view(-1)

        submatrix = torch.stack((v1, v2, v3, v4)).T

        return submatrix


    def generate_Q_matrix(self):
        """
        Q_{k, o_i, o_j} = Pr(Arriving in state k | Agent i observes o_i, Agent j observed o_j)
        """
        mat1 = self.generate_Q_submatrix([0, 1], [0, 2])
        mat2 = self.generate_Q_submatrix([1, 0], [1, 3])
        mat3 = self.generate_Q_submatrix([2, 3], [2, 0])
        mat4 = self.generate_Q_submatrix([3, 2], [3, 1])
#        return torch.stack((mat1, mat2, mat3, mat4))
        return torch.stack((mat1, mat2, mat3, mat4)).view(16, 4)




    def find_values(self, agent_index):
        
        I = torch.eye(4)  # Identity matrix of size 4x4
        self.subtracted_matrix = I - self.gamma * torch.matmul(self.P, self.Q)

        inverse_matrix = torch.linalg.solve(self.subtracted_matrix, torch.eye(4))
        agent_rewards = torch.flatten(self.game[agent_index])

        return torch.matmul(inverse_matrix, agent_rewards)

    def optimize_pi(self, num_iterations, learning_rate=0.05):
        logit_p_i = torch.log(self.p_i / (1 - self.p_i)).clone().detach().requires_grad_(True)  # Logit transformation

        optimizer = torch.optim.Adam([logit_p_i], lr=learning_rate)

        for i in range(num_iterations):
            #print(f"\n Run {i}")
            optimizer.zero_grad()

            self.p_i = torch.sigmoid(logit_p_i)
            self.Q = self.generate_Q_matrix()
            self.values_0 = self.find_values(0)
            self.values_1 = self.find_values(1)
                        # gotta sort this all out
            loss = -self.values_0.sum()  # We want to maximize self.values_0, so we negate it for minimization
            loss.backward(retain_graph=True)
            
            #print("p_i values:", self.p_i)
            #print("value for agent i:", self.values_0)
            #print("total value for agent i:", self.values_0.sum())
            #print("loss:", loss)  # Check the value of the loss
            optimizer.step()

#            with torch.no_grad():  # We don't want these operations to be tracked in the computational graph
#                eps = 1e-7
#                logit_p_i = torch.log((self.p_i + eps) / (1 - self.p_i + eps)).clone().detach().requires_grad_(True)


prisoners_dilemma = torch.tensor([[[3, 0],
                                   [4, 1]],
                                   
                                  [[3, 4],
                                   [0, 1]]])


if __name__ == "__main__":
    eps = 1e-7
    p = 0.01
    gamma = 0.95

    game_1 = NoisyIPD(game = prisoners_dilemma,
                            p_i = torch.tensor([0.1, 0.2, 0.3, 0.4]).float(),
                            p_j = torch.tensor([0.5, 0.6, 0.7, 0.8]).float(),
                            p = p,
                            gamma = gamma)
    game_1.optimize_pi(num_iterations=10000)
    game_1.p_i
