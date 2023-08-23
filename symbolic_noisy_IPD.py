import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sympy import symbols, Matrix, pprint, eye, simplify, diff, Float, Eq, solve, im, re
import time


def form_P_matrix(p):
    P = Matrix([
        [(1-p) ** 2, (1-p) * p, p * (1-p), p ** 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, (1-p) ** 2, (1-p) * p, p * (1-p), p ** 2, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, (1-p) ** 2, (1-p) * p, p * (1-p), p ** 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (1-p) ** 2, (1-p) * p, p * (1-p), p ** 2],
        ])
    return P

def form_Q_matrix(p_i, p_j):
    p_0 = p_i[0]
    p_1 = p_i[1]
    p_2 = p_i[2]
    p_3 = p_i[3]
    q_0 = p_j[0]
    q_1 = p_j[1]
    q_2 = p_j[2]
    q_3 = p_j[3]
    
    Q = Matrix([
        [(1-p_0)*(1-q_0), (1-p_0)*q_0, p_0*(1-q_0), p_0*q_0],
        [(1-p_0)*(1-q_2), (1-p_0)*q_2, p_0*(1-q_2), p_0*q_2],
        [(1-p_1)*(1-q_0), (1-p_1)*q_0, p_1*(1-q_0), p_1*q_0],
        [(1-p_1)*(1-q_2), (1-p_1)*q_2, p_1*(1-q_2), p_1*q_2],

        [(1-p_1)*(1-q_1), (1-p_1)*q_1, p_1*(1-q_1), p_1*q_1],
        [(1-p_1)*(1-q_3), (1-p_1)*q_3, p_1*(1-q_3), p_1*q_3],
        [(1-p_0)*(1-q_1), (1-p_0)*q_1, p_0*(1-q_1), p_0*q_1],
        [(1-p_0)*(1-q_3), (1-p_0)*q_3, p_0*(1-q_3), p_0*q_3],

        [(1-p_2)*(1-q_2), (1-p_2)*q_2, p_2*(1-q_2), p_2*q_2],
        [(1-p_2)*(1-q_0), (1-p_2)*q_0, p_2*(1-q_0), p_2*q_0],
        [(1-p_3)*(1-q_2), (1-p_3)*q_2, p_3*(1-q_2), p_3*q_2],
        [(1-p_3)*(1-q_0), (1-p_3)*q_0, p_3*(1-q_0), p_3*q_0],

        [(1-p_3)*(1-q_3), (1-p_3)*q_3, p_3*(1-q_3), p_3*q_3],
        [(1-p_3)*(1-q_1), (1-p_3)*q_1, p_3*(1-q_1), p_3*q_1],
        [(1-p_2)*(1-q_3), (1-p_2)*q_3, p_2*(1-q_3), p_2*q_3],
        [(1-p_2)*(1-q_1), (1-p_2)*q_1, p_2*(1-q_1), p_2*q_1],
    ])
    return Q


def form_det(a0, a1, a2, a3,
             b0, b1, b2, b3,
             c0, c1, c2, c3,
             d0, d1, d2, d3):
    print("Forming determinant")
    term1 = a0 * b1 * c2 * d3
    term2 = -a0 * b1 * c3 * d2
    term3 = -a0 * b2 * c1 * d3
    term4 = a0 * b2 * c3 * d1
    term5 = a0 * b3 * c1 * d2
    term6 = -a0 * b3 * c2 * d1
    term7 = -a1 * b0 * c2 * d3
    term8 = a1 * b0 * c3 * d2
    term9 = a1 * b2 * c0 * d3
    term10 = -a1 * b2 * c3 * d0
    term11 = -a1 * b3 * c0 * d2
    term12 = a1 * b3 * c2 * d0
    term13 = a2 * b0 * c1 * d3
    term14 = -a2 * b0 * c3 * d1
    term15 = -a2 * b1 * c0 * d3
    term16 = a2 * b1 * c3 * d0
    term17 = a2 * b3 * c0 * d1
    term18 = -a2 * b3 * c1 * d0
    term19 = -a3 * b0 * c1 * d2
    term20 = a3 * b0 * c2 * d1
    term21 = a3 * b1 * c0 * d2
    term22 = -a3 * b1 * c2 * d0
    term23 = -a3 * b2 * c0 * d1
    term24 = a3 * b2 * c1 * d0
    print("\tFinished forming determinant terms, now adding and simplifying")
    out = simplify(term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + \
          term9 + term10 + term11 + term12 + term13 + term14 + term15 + term16 + \
          term17 + term18 + term19 + term20 + term21 + term22 + term23 + term24)
    return out


def form_adj(a0, a1, a2, a3,
             b0, b1, b2, b3,
             c0, c1, c2, c3,
             d0, d1, d2, d3):
    print("Forming adjunct matrix")
    out = Matrix([
        [ b1*c2*d3 - b1*c3*d2 - b2*c1*d3 + b2*c3*d1 + b3*c1*d2 - b3*c2*d1, -a1*c2*d3 + a1*c3*d2 + a2*c1*d3 - a2*c3*d1 - a3*c1*d2 + a3*c2*d1,  a1*b2*d3 - a1*b3*d2 - a2*b1*d3 + a2*b3*d1 + a3*b1*d2 - a3*b2*d1, -a1*b2*c3 + a1*b3*c2 + a2*b1*c3 - a2*b3*c1 - a3*b1*c2 + a3*b2*c1],
        [-b0*c2*d3 + b0*c3*d2 + b2*c0*d3 - b2*c3*d0 - b3*c0*d2 + b3*c2*d0,  a0*c2*d3 - a0*c3*d2 - a2*c0*d3 + a2*c3*d0 + a3*c0*d2 - a3*c2*d0, -a0*b2*d3 + a0*b3*d2 + a2*b0*d3 - a2*b3*d0 - a3*b0*d2 + a3*b2*d0,  a0*b2*c3 - a0*b3*c2 - a2*b0*c3 + a2*b3*c0 + a3*b0*c2 - a3*b2*c0],
        [ b0*c1*d3 - b0*c3*d1 - b1*c0*d3 + b1*c3*d0 + b3*c0*d1 - b3*c1*d0, -a0*c1*d3 + a0*c3*d1 + a1*c0*d3 - a1*c3*d0 - a3*c0*d1 + a3*c1*d0,  a0*b1*d3 - a0*b3*d1 - a1*b0*d3 + a1*b3*d0 + a3*b0*d1 - a3*b1*d0, -a0*b1*c3 + a0*b3*c1 + a1*b0*c3 - a1*b3*c0 - a3*b0*c1 + a3*b1*c0],
        [-b0*c1*d2 + b0*c2*d1 + b1*c0*d2 - b1*c2*d0 - b2*c0*d1 + b2*c1*d0,  a0*c1*d2 - a0*c2*d1 - a1*c0*d2 + a1*c2*d0 + a2*c0*d1 - a2*c1*d0, -a0*b1*d2 + a0*b2*d1 + a1*b0*d2 - a1*b2*d0 - a2*b0*d1 + a2*b1*d0,  a0*b1*c2 - a0*b2*c1 - a1*b0*c2 + a1*b2*c0 + a2*b0*c1 - a2*b1*c0]])
    print("\tFinished summing adjunct matrix, now simplifying")
    return simplify(out)


def form_inv(a0, a1, a2, a3,
             b0, b1, b2, b3,
             c0, c1, c2, c3,
             d0, d1, d2, d3):
    det = form_det(a0, a1, a2, a3,
             b0, b1, b2, b3,
             c0, c1, c2, c3,
             d0, d1, d2, d3)
    adj = form_adj(a0, a1, a2, a3,
             b0, b1, b2, b3,
             c0, c1, c2, c3,
             d0, d1, d2, d3)
    return adj / det

def round_expr(expr, num_digits):
    return expr.xreplace({n: round(n, num_digits) for n in expr.atoms(Float)})


class NoisyIPDSolver:
    def __init__(self, p, p_i, p_j, gamma, rewards):
        self.p = p 
        self.p_i = p_i
        self.p_j = p_j
        self.gamma = gamma
        self.rewards = rewards
        self.P = form_P_matrix(self.p)
        self.Q = form_Q_matrix(self.p_i, self.p_j)
        self.mat = simplify(eye(4) - gamma * self.P * self.Q)
        self.det = form_det(self.mat[0], self.mat[1], self.mat[2], self.mat[3],
                                self.mat[4], self.mat[5], self.mat[6], self.mat[7],
                                self.mat[8], self.mat[9], self.mat[10], self.mat[11],
                                self.mat[12], self.mat[13], self.mat[14], self.mat[15],
                                )
        self.adj = form_adj(self.mat[0], self.mat[1], self.mat[2], self.mat[3],
                                self.mat[4], self.mat[5], self.mat[6], self.mat[7],
                                self.mat[8], self.mat[9], self.mat[10], self.mat[11],
                                self.mat[12], self.mat[13], self.mat[14], self.mat[15],
                                )
        self.inv = self.adj / self.det
        self.original_vals = simplify(self.inv * self.rewards)
#        self.original_vals = simplify(self.adj * self.rewards)
        #self.original_vals = self.original_vals.applyfunc(lambda x: round_expr(x, 10))


    def find_optimal_pi(self):
        """
        Finds p_i that maximises the value of the first node (L, L)
        Unfinished
        """
        deriv_0 = diff(self.original_vals[0], self.p_i[0])
        deriv_1 = diff(self.original_vals[0], self.p_i[1])
        deriv_2 = diff(self.original_vals[0], self.p_i[2])
        deriv_3 = diff(self.original_vals[0], self.p_i[3])
        
        equations = [
            Eq(deriv_0, 0),
            Eq(deriv_1, 0),
            Eq(deriv_2, 0),
            Eq(deriv_3, 0)
        ]
        solve(equations, self.p_i)


def find_crossovers(minmax_obj, collusive_obj):

    minmax_vals = minmax_obj.original_vals
    coll_vals = collusive_obj.original_vals

    minmax_sum = minmax_vals[0] + minmax_vals[1] + minmax_vals[2] + minmax_vals[3]
    coll_sum = coll_vals[0] + coll_vals[1] + coll_vals[2] + coll_vals[3]

    sum_diff = simplify(minmax_sum - coll_sum)
    solution = solve(sum_diff, p)
    simplified_solution = [simplify(i) for i in solution]
    real_solution = [i for i in simplified_solution if im(simplify(i.subs(gamma, 0.5))) == 0][0]
    return simplify(real_solution)



p = symbols('p', real=True)
#p = 0.3
gamma = symbols('gamma', real=True)
#gamma = 0.95

# rewards = Matrix([r0, r1, r2, r3])
# p_0, p_1, p_2, p_3 = symbols('pi_LL, pi_LC, pi_CL, pi_CC')
# q_0, q_1, q_2, q_3 = symbols('pj_LL, pj_LC, pj_CL, pj_CC')
r0, r1, r2, r3 = symbols('r0, r1, r2, r3')

def evaluate_solution(sol, val):
    if im(sol.subs(gamma, val)) != 0:
        print(f"Imaginary part non-zero for gamma = {val}")
    return float(re(sol.subs(gamma, val)))


def save_plot_IPD_crossovers(rewards):
    print(f"Building plot for following rewards: {rewards}")

    p_i = Matrix([1, 1, 1, 1])
    p_j = Matrix([0, 1, 1, 1])
    strat1minmax_obj = NoisyIPDSolver(p, p_i, p_j, gamma, rewards)

    p_i = Matrix([0, 1, 1, 1])
    p_j = Matrix([0, 1, 1, 1])
    strat1_obj = NoisyIPDSolver(p, p_i, p_j, gamma, rewards)

    p_i = Matrix([1, 1, 1, 1])
    p_j = Matrix([0, 1, 1, 0])
    strat2minmax_obj = NoisyIPDSolver(p, p_i, p_j, gamma, rewards)

    p_i = Matrix([0, 1, 1, 0])
    p_j = Matrix([0, 1, 1, 0])
    strat2_obj = NoisyIPDSolver(p, p_i, p_j, gamma, rewards)

    p_i = Matrix([1, 1, 1, 1])
    p_j = Matrix([0, 0, 1, 1])
    tft_minmax_obj = NoisyIPDSolver(p, p_i, p_j, gamma, rewards)

    p_i = Matrix([0, 1, 0, 1])
    p_j = Matrix([0, 0, 1, 1])
    tft_obj = NoisyIPDSolver(p, p_i, p_j, gamma, rewards)

    print("Finished constructing objects")
    print("Calculating symbolic crossovers 1")
    s1_sol = find_crossovers(strat1minmax_obj, strat1_obj)
    print("Calculating symbolic crossovers 2")
    s2_sol = find_crossovers(strat2minmax_obj, strat2_obj)
    print("Calculating symbolic crossovers TFT")
    tft_sol = find_crossovers(tft_minmax_obj, tft_obj)

    step_size = 0.001
    gamma_vals = np.arange(step_size, 1, step_size)
    print("Calculating numeric crossovers 1")
    s1_vals = [evaluate_solution(s1_sol, val) for val in gamma_vals]
    print("Calculating numeric crossovers 2")
    s2_vals = [evaluate_solution(s2_sol, val) for val in gamma_vals]
    print("Calculating numeric crossovers TFT")
    tft_vals = [evaluate_solution(tft_sol, val) for val in gamma_vals]

    vals_full = pd.DataFrame({"Strategy 1": s1_vals,
                              "Strategy 2": s2_vals,
                              "TFT": tft_vals}, index=gamma_vals)

    for column in vals_full.columns:
        plt.plot(vals_full[column].apply(lambda x: max(x, 0)), label=column)

    plt.title(f"Crossovers with rewards: {[i for i in rewards]}")
    plt.legend()
    plt.savefig(f"crossover_plots/{rewards[0]}_{rewards[1]}_{rewards[2]}_{rewards[3]}.jpg")
    plt.close()



rewards_0 = Matrix([2, 0, 3, 1])
rewards_1 = Matrix([3, 0, 4, 1])
rewards_2 = Matrix([2, 0, 10, 1])
rewards_3 = Matrix([2, 0, 100, 1])
rewards_4 = Matrix([2, 0, 1000, 1])
rewards_5 = Matrix([100, 0, 101, 1])
rewards_6 = Matrix([1000, 0, 1001, 1])
rewards_7 = Matrix([10000, 0, 10001, 1])
rewards_8 = Matrix([10000000, 0, 10000001, 1])
rewards_9 = Matrix([6, 0, 8, 2])


for rewards in [rewards_0, rewards_1, rewards_2, rewards_3,
                rewards_4, rewards_5, rewards_6, rewards_7, 
                rewards_8, rewards_9]:
    save_plot_IPD_crossovers(rewards)
