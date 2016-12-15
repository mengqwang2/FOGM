import numpy as np
from numpy.linalg import inv
from scipy.stats import wishart
from scipy.stats import bernoulli
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm

K = 5
V = 7
N = 10
D = 20
T = 3

sigma = 3
alpha = 2 ** np.random.randint(0, 4, size=(K))
beta_0 = 2 ** np.random.randint(0, 3, size=(K, V))

def generate_dirichlet(alpha):
    return np.random.dirichlet(alpha)

def generate_multinomial(theta):
    z_prob = np.random.multinomial(1, theta)
    for ind in range(len(z_prob)):
        if z_prob[ind] == 1:
            return ind

def generate_mvn(mean):
    beta_new = np.empty(mean.shape)
    for i in range(mean.shape[0]):
        beta_new[i] = np.random.multivariate_normal(mean[i], sigma * np.identity(V))
    return beta_new

def normalizer_pi(beta):
    beta_normalized = np.empty(beta.shape)
    for k in range(beta.shape[0]):
        s = 0
        for i in range(beta.shape[1]):
            s += np.exp(beta[k][i])
        for i in range(beta.shape[1]):
            beta_normalized[k][i] = np.exp(beta[k][i]) * 1.0 / s
    return beta_normalized

if __name__ == "__main__":
    beta = np.empty((T, K, V))
    theta = np.empty(K)
    z = np.empty(N)
    beta[0] = beta_0
    document = np.empty((D, N))
    for t in range(1, T):
        # generate beta (K * V)
        beta[t] = generate_mvn(beta[t - 1])
        beta[t] = normalizer_pi(beta[t])
        for d in range(D):
            theta = generate_dirichlet(alpha)
            for n in range(N):
                z = generate_multinomial(theta)
                word = generate_multinomial(beta[t][z])
                document[d][n] = word
    print document



