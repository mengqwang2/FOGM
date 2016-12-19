import numpy as np
from numpy.linalg import inv
from scipy.stats import wishart
from scipy.stats import bernoulli
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
from scipy import misc
from scipy import special as sp
import time


K = 5
V = 5352
D = 2078
T = 5
N = 146

S = 500 #the number of samples for approximating E(log sum exp on beta) each time
iteration = 200
alpha = np.zeros((T, D, K)) # topic proportions for each word in each document
z = np.zeros((T, D, N)) # topic assignment for each word
phi = np.zeros((T, D, N, K)) 
beta = np.zeros((T, K, V)) # word proportions for each topic
mu_beta_t = np.zeros((T, K, V))
cov_beta_t = np.zeros((T, K, V, V))

#store this expectation of log sum exp in a TxK array
E_log_sum_exp_beta_t_k = np.zeros((T,K))


alpha_0 = 1
sigma = 1
#alpha = np.random.randint(0, 4, size=(K))
beta_0 = np.zeros((K, V))
document = np.load('document_real.npy') # (T, N, D)
voc_map = np.load('voc.npy')
D_N = np.load('word_count.npy')
phi = np.zeros((T, D, N, K))

for t in range(T):
    for d in range(D):
        for n in range(N):
            phi[t][d][n] = np.random.dirichlet(np.ones(K))

def normalize(arr):
    s = 0
    for x in arr:
        s += x
    return np.asarray([x * 1.0 / s for x in arr])

def update_alpha():
    global alpha
    for t in range(T):
        for d in range(D):
            sum_phi = np.zeros((K))
            for n in range(int(D_N[t][d])):
                sum_phi += phi[t][d][n]
            alpha[t][d] = alpha_0 + sum_phi
            # normalize 
            # alpha[t][d] = normalize(alpha[t][d])

'''
def t_func(beta_k):
    voc_sum = 0
    for v in range(V):
        voc_sum += np.exp(beta_k[v])
    return voc_sum
'''

def s_func(beta_k):
    return np.asarray([np.exp(beta_k[v]) for v in range(V)])

def B_func(phi, k, t):
    B = np.zeros((V))
    for d in range(D):
        for n in range(int(D_N[t][d])):
            B[int(document[t][d][n])] += phi[t][d][n][k]
    return B

def gradient_descent_beta_0(k):
    step = 0.5
    precision = 0.1
    voc_sum = 0
    old = np.asarray([0 for _ in range(V)])
    new = old
    ite = 1

    while 1:
        #voc_sum = t_func(old)
        exp_beta = s_func(old)
        voc_sum = sum(exp_beta)
        delta_g = np.asarray([exp_beta[v] / voc_sum for v in range(V)])
        b_val = B_func(phi, k, 0)
        ## x = -1.0 * old - 1.0 / (sigma ** 2) * old + 1.0 / (sigma ** 2) * mu_beta_t[1][k] - delta_g
        x = -1.0 * old - 1.0 / (sigma ** 2) * old + 1.0 / (sigma ** 2) * mu_beta_t[1][k] + b_val - delta_g * sum(b_val)
        new = old + step * x / ite
        #cond = 1
        #for v in range(V):
        #    if new[v] - old[v] > precision:
        #        cond = 0
        #        break
        #if cond:
        #    break
        old = new
        if np.dot(x, x) < precision:
            break
        ite += 1
    return new


def update_beta_0():
    global mu_beta_t
    global cov_beta_t
    for k in range(K):
        mu_beta_t[0][k] = gradient_descent_beta_0(k)
 
        s_func_val = s_func(mu_beta_t[0][k])
        t_func_val = sum(s_func_val)
        b_val = B_func(phi, k, 0)
        #diag = np.zeros((V, V))
        #for v in range(V):
        #    diag[v][v] = s_func_val[v]
        diag=np.diag(s_func_val)
        delta_sqr = np.identity(V) + 1.0 / (sigma ** 2) * np.identity(V) + sum(b_val) * (t_func_val * diag - np.outer(s_func_val, s_func_val)) / (t_func_val ** 2)
        
        A = np.ones(V) * (1 + 1 / (sigma ** 2) ) + sum(b_val) * (t_func_val * np.diag(diag)) / (t_func_val ** 2) 
        Ainv = 1 / A  #array
        c = np.sqrt(sum(b_val)) / t_func_val
        v = c * s_func_val
        downstair = 1 - sum(Ainv * v * v)
        AinvV = Ainv * v
        upstair = np.outer(AinvV, AinvV)
        np.diag(Ainv) + upstair/downstair
        cov_beta_t[0][k] = np.diag(Ainv) + upstair/downstair

        #cov_beta_t[0][k] = np.linalg.inv(delta_sqr)
        #beta[0][k] = np.random.multivariate_normal(mu_beta_t[0][k], cov_beta_t[0][k])


def gradient_descent_beta_t(k, t):
    step = 0.5
    precision = 0.1
    voc_sum = 0
    old = np.asarray([0 for _ in range(V)])
    new = old
    ite = 1

    while 1:
        #voc_sum = t_func(old)
        exp_beta = s_func(old)
        voc_sum = sum(exp_beta)
        delta_g = np.asarray([exp_beta[v] / voc_sum for v in range(V)])
        b_val=B_func(phi, k, t)
        x = -2.0 / (sigma ** 2) * old + 1.0 / (sigma ** 2) * mu_beta_t[t + 1][k] + 1.0 / (sigma ** 2) * mu_beta_t[t - 1][k] +b_val-delta_g*sum(b_val)
        new = old + step * x / ite
        #cond = 1
        #for v in range(V):
        #    if new[v] - old[v] > precision:
        #        cond = 0
        #        break
        #if cond:
        #    break
        old = new
        if np.dot(x,x) < precision:
            break
        ite += 1
    return new

def update_beta_t():
    global beta
    global mu_beta_t
    global cov_beta_t

    for t in range(1, T - 1):
        for k in range(K):
            mu_beta_t[t][k] = gradient_descent_beta_t(k, t)
            s_func_val = s_func(mu_beta_t[t][k])
            t_func_val = sum(s_func_val)
            
            b_val = B_func(phi, k, t)
            #diag = np.zeros((V, V))
            #for v in range(V):
            #    diag[v][v] = s_func_val[v]
            diag=np.diag(s_func_val)
            A=np.ones(V)*(2 / (sigma ** 2) )+ sum(b_val) * (t_func_val * np.diag(diag))/ (t_func_val ** 2) 
            Ainv=1/A  #array
            c=np.sqrt(sum(b_val))/t_func_val
            v=c*s_func_val
            downstair=1-sum(Ainv*v*v)
            AinvV=Ainv*v
            upstair=np.outer(AinvV,AinvV)
            cov_beta_t[t][k]=np.diag(Ainv)+upstair/downstair
            #delta_sqr = 2.0 / (sigma ** 2) * np.identity(V) +sum(b_val) * (t_func_val * diag - np.outer(s_func_val, s_func_val)) / (t_func_val ** 2)
            #cov_beta_t[t][k] = np.linalg.inv(delta_sqr)
            #beta[t][k] = np.random.multivariate_normal(mu_beta_t[t][k], cov_beta_t[t][k])

def gradient_descent_beta_T(k):
    step = 0.5
    precision = 0.1
    voc_sum = 0
    old = np.asarray([0 for _ in range(V)])
    new = old
    ite = 1

    while 1:
        #voc_sum = t_func(old)
        exp_beta = s_func(old)
        voc_sum = sum(exp_beta)
        delta_g = np.asarray([exp_beta[v] / voc_sum for v in range(V)])
        b_val=B_func(phi, k, T - 1)
        x = -1.0 / (sigma ** 2) * old + 1.0 / (sigma ** 2) * mu_beta_t[T - 2][k] + b_val - delta_g * sum(b_val)
        new = old + step * x / ite
        #cond = 1
        #for v in range(V):
        #    if new[v] - old[v] > precision:
        #        cond = 0
        #        break
        #if cond:
        #    break
        old = new
        if np.dot(x, x) < precision:
            break
        ite += 1
    return new

def update_beta_T():
    global mu_beta_t
    global cov_beta_t

    for k in range(K):
        mu_beta_t[T - 1][k] = gradient_descent_beta_T(k)
       
        s_func_val = s_func(mu_beta_t[T - 1][k])
        t_func_val = sum(s_func_val)
        b_val = B_func(phi, k, T - 1)
        #diag = np.zeros((V, V))
        #for v in range(V):
        #    diag[v][v] = s_func_val[v]
        diag = np.diag(s_func_val)
        #delta_sqr = 1.0 / (sigma ** 2) * np.identity(V) + sum(b_val) * (t_func_val * diag - np.outer(s_func_val, s_func_val)) / (t_func_val ** 2)
        #cov_beta_t[T - 1][k] = np.linalg.inv(delta_sqr)
        #beta[T - 1][k] = np.random.multivariate_normal(mu_beta_t[T - 1][k], cov_beta_t[T - 1][k])
        A=np.ones(V)*(1 + 1.0 / (sigma ** 2) )+ sum(b_val) * (t_func_val * np.diag(diag))/ (t_func_val ** 2) 
        #vvt=- sum(b_val)*np.outer(s_func_val, s_func_val) / (t_func_val ** 2)
        Ainv=1/A  #array
        c=np.sqrt(sum(b_val))/t_func_val
        v=c*s_func_val
        downstair=1-sum(Ainv*v*v)
        AinvV=Ainv*v
        upstair=np.outer(AinvV,AinvV)
        cov_beta_t[T - 1][k]=np.diag(Ainv)+upstair/downstair


def update_EXP_log_sum_exp_beta_t_k():
    global E_log_sum_exp_beta_t_k
    for t in range(T):
        for k in range(K):
            samples_sum=0
            for s in range(S):
                #beta_s=np.random.multivariate_normal(mu_beta_t[t][k], cov_beta_t[t][k])
                x=np.random.normal(0, 1, V)
                lambda_=np.sqrt(np.diag(cov_beta_t[t][k]))
                beta_s=lambda_*x+mu_beta_t[t][k]
                samples_sum+=misc.logsumexp(x)
            E_log_sum_exp_beta_t_k[t][k]=samples_sum/S

def update_phi():
    global phi
    for t in range(T):
        for d in range(D):
            quick_digamma = sp.digamma(sum(alpha[t][d]))  #this is the same for every word n on every topic k
            for n in range(int(D_N[t][d])):
                phi_temp = np.zeros(K)
                log_phi_temp = np.zeros(K)
                word_index = int(document[t][d][n])
                for k in range(K):
                    p1 = sp.digamma(alpha[t][d][k]) - quick_digamma  #log of the contribution from theta
                    p2 = mu_beta_t[t][k][word_index] - E_log_sum_exp_beta_t_k[t][k] #log of the contribution from beta
                    log_phi_temp[k] = p1 + p2
                max_log = max(log_phi_temp)
                log_phi_temp = log_phi_temp - max_log
                phi_temp = normalize(np.exp(log_phi_temp))
                phi[t][d][n] = phi_temp

###ELBO###
def E_log_p_beta_0():
    summ = 0
    for k in range(K):
        ##Gaussian quadratic form identity
        temp = np.dot(mu_beta_t[0][k], mu_beta_t[0][k]) + sum(np.diag(cov_beta_t[0][k]))
        summ -= 0.5 * temp
    return summ

def E_log_p_beta_t():
    summ=0
    for t in range(1, T):
        for k in range(K):
            temp1 = np.dot(mu_beta_t[t][k],mu_beta_t[t][k]) + sum(np.diag(cov_beta_t[t][k]))
            temp2 = np.dot(mu_beta_t[t - 1][k],mu_beta_t[t - 1][k]) + sum(np.diag(cov_beta_t[t - 1][k]))
            temp3 = (-2) * np.dot(mu_beta_t[t][k],mu_beta_t[t - 1][k])
            summ -= 1 / (2 * sigma ** 2) * (temp1 + temp2 + temp3)
    return summ

def E_log_p_theta():
    summ=0
    for t in range(T):
        for d in range(D):
            alpha_t_d=alpha[t][d]
            E_log_theta=np.array(sp.digamma(alpha_t_d))-sp.digamma(sum(alpha_t_d))
            temp=np.dot(np.ones(K)*(alpha_0-1),E_log_theta)
            summ+=temp
    return summ


def E_log_p_z_and_w_cond_beta_theta():
    ##first get K-array for the digamma on sum of gamma_k
    total=0
    for t in range(T):
        for d in range(D):
            for n in range(int(D_N[t][d])):
                word_index=int(document[t][d][n])

                v2=sp.digamma(alpha[t][d])-sp.digamma(sum(alpha[t][d]))+np.transpose(mu_beta_t[t])[word_index]-E_log_sum_exp_beta_t_k[t][k]
                total+=np.dot(phi[t][d][n],v2)
    #print "E done"
    return total

def entropy_beta():
    summ=0
    for t in range(T):
        for k in range(K):
            temp=0.5*np.linalg.det((2*np.pi*np.exp(1)*cov_beta_t[t][k]))
            summ+=temp
    return summ

##another way to compute the entropy on beta that could be faster###
def entropy_beta_quick():
    const=2*np.pi*np.exp(1)
    summ=0
    for t in range(T):
        for k in range(K):
            temp=0.5*(const**V)*np.exp(sum(np.log(np.diag(cov_beta_t[t][k]))))
            summ+=temp
    return summ
## another way to compute the entropy on beta that could be faster###

def entropy_theta():
    summ=0
    for t in range(T):
        for d in range(D):
            alpha_t_d=alpha[t][d]
            kk=len(alpha_t_d)
            alpha_sum=sum(alpha_t_d)
            h=sum(sp.gammaln(alpha_t_d))-sp.gammaln(alpha_sum)
            h+=(alpha_sum-kk)*sp.digamma(alpha_sum)
            h-=np.dot(alpha_t_d-1,sp.digamma(alpha_t_d))
            summ+=h
    return summ

def entropy_phi():
    h=0
    for t in range(T):
        for d in range(D):
            for n in range(int(D_N[t][d])):
                h+=-np.dot(np.log(phi[t][d][n]),phi[t][d][n])
        #if d%10000==0:
        #    print d
    return h

def ELBO():
    global alpha
    global mu_beta_t
    global cov_beta_t
    global phi
    global E_log_sum_exp_beta_t_k
    elbo = 0
    ##first everything from nominator
    elbo += E_log_p_beta_0()

    elbo += E_log_p_beta_t()

    elbo += E_log_p_theta()

    elbo += E_log_p_z_and_w_cond_beta_theta()

    ##then everything from denomiator (entropy)
    elbo -= entropy_beta_quick()

    elbo -= entropy_theta()

    elbo -= entropy_phi()

    return elbo

###end of ELBO section###


if __name__ == "__main__":
    elb = []
    for t in range(T):
        for d in range(D):
            for n in range(int(D_N[t][d])):
                k = np.random.randint(K)
                phi[t][d][n][k] = 1
    for ite in range(iteration):
        # update beta_0
        #start = time.time()
        update_beta_0()
        #end = time.time()
        #print end - start
        # update beta_t
        update_beta_t()
        # update beta_{T-1}
        update_beta_T()
        # update theta
        update_alpha()
        # update expectation
        update_EXP_log_sum_exp_beta_t_k()
        # update z
        update_phi()

        for t in range(T):
            for k in range(K):
                beta[t][k] = np.random.multivariate_normal(mu_beta_t[t][k], cov_beta_t[t][k])

        cur_elb = ELBO()
        elb.append(cur_elb)
        print cur_elb

    xaxis = [i for i in range(1, iteration + 1)]
    plt.plot(xaxis, elb)
    plt.xlabel("Iterations")
    plt.ylabel("ELBO")
    plt.title("ELBO change in variational inference")
    plt.show()

    np.save("beta.npy", beta)
    np.save("elbo.npy", np.asarray(elb))




