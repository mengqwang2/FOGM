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
iteration = 100
theta = np.zeros((T, D, K)) # topic proportions for each word in each document
z = np.zeros((T, D, N)) # topic assignment for each word
phi = np.zeros((T, D, N, K)) # indicator vector for each topic assignment 
beta = np.zeros((T, K, V)) # word proportions for each topic
mu_beta_t = np.zeros((T, K, V))
cov_beta_t = np.zeros((T, K, V, V))

sigma = 1
alpha = np.random.randint(0, 4, size=(K))
beta_0 = np.random.randint(0, 3, size=(K, V))
document = np.load('document.npy') # (T, D, N)

def normalize(arr):
    s = 0
    for x in arr:
        s += x
    return np.asarray([x * 1.0 / s for x in arr])

def update_theta():
    global theta
    for t in range(T):
        for d in range(D):
            sum_phi = np.zeros((K))
            for n in range(N):
                sum_phi += phi[t][d][n]
            theta[t][d] = sum_phi
            # normalize
            theta[t][d] = normalize(theta[t][d])

def t_func(beta_k):
    voc_sum = 0
    for v in range(V):
        voc_sum += np.exp(beta_k[v])
    return voc_sum

def s_func(beta_k):
    return np.asarray([np.exp(beta_k[v]) for v in range(V)])

def B_func(phi, k, t):
    B = np.zeros((V))
    for d in range(D):
        for n in range(N):
            B[int(document[t][d][n])] += phi[t][d][n][k]
    return B

def gradient_descent_beta_0(k):
    step = 0.1
    precision = 0.1
    voc_sum = 0
    old = np.asarray([0 for _ in range(V)])
    new = old

    while 1:
        voc_sum = t_func(old)
        exp_beta = s_func(old)
        delta_g = np.asarray([exp_beta[v] / voc_sum for v in range(V)])
        x = -1.0 * old - 1.0 / (sigma ** 2) * old + 1.0 / (sigma ** 2) * mu_beta_t[1][k] - delta_g
        new = old - step * x
        cond = 1
        for v in range(V):
            if new[v] - old[v] > precision:
                cond = 0
                break
        if cond:
            break
        old = new
    return new


def update_beta_0():
    global beta
    global mu_beta_t
    global cov_beta_t
    for k in range(K):
        mu_beta_t[0][k] = gradient_descent_beta_0(k)
        t_func_val = t_func(mu_beta_t[0][k])
        s_func_val = s_func(mu_beta_t[0][k])
        b_val = B_func(phi, k, 0)
        diag = np.zeros((V, V))
        for v in range(V):
            diag[v][v] = s_func_val[v]
        delta_sqr = np.identity(V) + 1.0 / (sigma ** 2) * np.identity(V) + sum(b_val) * (t_func_val * diag - np.dot(s_func_val.transpose(), s_func_val)) / (t_func_val ** 2)
        cov_beta_t[0][k] = np.linalg.inv(delta_sqr)
        beta[0][k] = np.random.multivariate_normal(mu_beta_t[0][k], cov_beta_t[0][k])

def gradient_descent_beta_t(k, t):
    step = 0.1
    precision = 0.1
    voc_sum = 0
    old = np.asarray([0 for _ in range(V)])
    new = old

    while 1:
        voc_sum = t_func(old)
        exp_beta = s_func(old)
        delta_g = np.asarray([exp_beta[v] / voc_sum for v in range(V)])
        x = -2.0 / (sigma ** 2) * old + 1.0 / (sigma ** 2) * mu_beta_t[t + 1][k] + 1.0 / (sigma ** 2) * mu_beta_t[t - 1][k] - delta_g
        new = old - step * x
        cond = 1
        for v in range(V):
            if new[v] - old[v] > precision:
                cond = 0
                break
        if cond:
            break
        old = new
    return new

def update_beta_t():
    global beta
    global mu_beta_t
    global cov_beta_t

    for t in range(1, T - 1):
        for k in range(K):
            mu_beta_t[t][k] = gradient_descent_beta_t(k, t)
            t_func_val = t_func(mu_beta_t[t][k])
            s_func_val = s_func(mu_beta_t[t][k])
            b_val = B_func(phi, k, t)
            diag = np.zeros((V, V))
            for v in range(V):
                diag[v][v] = s_func_val[v]
            delta_sqr = sum(b_val) * (t_func_val * diag - np.dot(s_func_val.transpose(), s_func_val)) / (t_func_val ** 2)
            cov_beta_t[t][k] = np.linalg.inv(delta_sqr)
            beta[t][k] = np.random.multivariate_normal(mu_beta_t[t][k], cov_beta_t[t][k])

def gradient_descent_beta_T(k):
    step = 0.1
    precision = 0.1
    voc_sum = 0
    old = np.asarray([0 for _ in range(V)])
    new = old

    while 1:
        voc_sum = t_func(old)
        exp_beta = s_func(old)
        delta_g = np.asarray([exp_beta[v] / voc_sum for v in range(V)])
        x = -2.0 / (sigma ** 2) * old + 1.0 / (sigma ** 2) * mu_beta_t[T - 2][k] - delta_g
        new = old - step * x
        cond = 1
        for v in range(V):
            if new[v] - old[v] > precision:
                cond = 0
                break
        if cond:
            break
        old = new
    return new

def update_beta_T():
    global beta
    global mu_beta_t
    global cov_beta_t

    for k in range(K):
        mu_beta_t[T - 1][k] = gradient_descent_beta_T(k)
        t_func_val = t_func(mu_beta_t[T - 1][k])
        s_func_val = s_func(mu_beta_t[T - 1][k])
        b_val = B_func(phi, k, T - 1)
        diag = np.zeros((V, V))
        for v in range(V):
            diag[v][v] = s_func_val[v]
        delta_sqr = sum(b_val) * (t_func_val * diag - np.dot(s_func_val.transpose(), s_func_val)) / (t_func_val ** 2)
        cov_beta_t[T - 1][k] = np.linalg.inv(delta_sqr)
        beta[T - 1][k] = np.random.multivariate_normal(mu_beta_t[T - 1][k], cov_beta_t[T - 1][k])

if __name__ == "__main__":
    #document = np.load('document.npy') # (T, N, D)
    phi = np.random.randint(0, 5, size=((T, D, N, K))) 
    for ite in range(iteration):
        # update theta
        update_theta()

        # update beta_0
        update_beta_0()

        # update beta_t
        update_beta_t()

        # update beata_{T-1}
        update_beta_T()

        # update z

