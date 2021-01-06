import os
import numpy as np
from scipy import stats
import numba
import math
import matplotlib
import matplotlib.pyplot as plt
import corner
from model import f_Rong, Df_Rong, model_Rong_init, model_space, Newton
from matplotlib import rc
rc('font',**{'family':'sans-serif','serif':['Computer Modern Roman'],'sans-serif':['Helvetica']})
rc('text', usetex=True)
matplotlib.rcParams['text.usetex'] = True

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
'''numerical simulation'''
def plot_chains(likelihood, annealing_t, chain, m_true, number_of_samples):
    fig0 = plt.figure(0)
    ax0a = plt.subplot(521)
    ax0a.plot(np.zeros((1,)), np.exp(likelihood[0]))
    ax0a.set_ylim(0,1.0)
    ax0a.set_xlabel('sample number')
    ax0a.set_ylabel('likelihood')
    ax0b = plt.subplot(522)
    ax0b.plot(np.zeros((1,)), annealing_t[0])
    ax0b.set_ylim(0,T_0)
    ax0b.set_xlabel('sample number')
    ax0b.set_ylabel('Temperatur')
    ax1 = plt.subplot(523)
    ax1.plot(chain[0,0], np.zeros((1,)))
    ax1.plot(m_true[0]*np.ones((number_of_samples,)), range(number_of_samples),'red')
    ax1.set_xlabel('lambda []')
    ax1.set_ylabel('sample number')
    ax2 = plt.subplot(524)
    ax2.plot(chain[0, 1], np.zeros((1,)))
    ax2.plot(m_true[1] * np.ones((number_of_samples,)), range(number_of_samples),'red')
    ax2.set_xlabel('gamma []')
    ax3 = plt.subplot(525)
    ax3.plot(chain[0, 2], np.zeros((1,)))
    ax3.plot(m_true[2] * np.ones((number_of_samples,)), range(number_of_samples),'red')
    ax3.set_xlabel('ks []')
    ax3.set_ylabel('sample number')
    ax4 = plt.subplot(526)
    ax4.plot(chain[0, 3], np.zeros((1,)))
    ax4.plot(m_true[3] * np.ones((number_of_samples,)), range(number_of_samples),'red')
    ax4.set_xlabel('kr []')
    ax5 = plt.subplot(527)
    ax5.plot(chain[0, 4], np.zeros((1,)))
    ax5.plot(m_true[4] * np.ones((number_of_samples,)), range(number_of_samples),'red')
    ax5.set_xlabel('delta []')
    ax5.set_ylabel('sample number')
    ax6 = plt.subplot(528)
    ax6.plot(chain[0, 5], np.zeros((1,)))
    ax6.plot(m_true[5] * np.ones((number_of_samples,)), range(number_of_samples),'red')
    ax6.set_xlabel('Ns []')
    ax7 = plt.subplot(529)
    ax7.plot(chain[0, 6], np.zeros((1,)))
    ax7.plot(m_true[6] * np.ones((number_of_samples,)), range(number_of_samples),'red')
    ax7.set_xlabel('c []')
    ax7.set_ylabel('sample number')
    ax8 = plt.subplot(5,2,10)
    ax8.plot(chain[0, 7], np.zeros((1,)))
    ax8.plot(m_true[7] * np.ones((number_of_samples,)), range(number_of_samples),'red')
    ax8.set_xlabel('Nr []')
    plt.subplots_adjust(hspace=0.9, wspace=0.5)
    plt.show()
    return ax0a, ax0b, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8

def forward_model(m=None,sim_info=None):
    assert m is not None, "Please define parameters."
    m = np.exp(m)
    assert sim_info is not None, "Please define simulation parameters."
    Nt = sim_info['Nt']
    h = sim_info['h']
    sampling_rate = sim_info['sampling_rate']
    u = np.zeros((5, Nt))
    _, drugs, u[:,0] = model_Rong_init()
    for i in range(1, Nt):
        x_old = u[:, i - 1] + h * f_Rong(u[:, i - 1], p=m, drugs=drugs)
        x_new = Newton(x_old, u[:, i-1], p=m, drugs=drugs, h=h)
        k = 0
        dev = np.abs(x_old - x_new)
        while dev[0] > tol and dev[1] > tol and dev[2] > tol:
            x_old = x_new
            x_new = Newton(x_old, u[:, i-1], p=m, drugs=drugs, h=h)
            dev = np.abs(x_old - x_new)
            k += 1
        u[:, i] = x_new
        if not (i % 10000):
            print('Solved at time ', str(i), ' after ', str(k), ' Newton iterations.')
        idx_obs = np.where(t%sampling_rate == 0)[0].astype(int)[:-1]
    return u[:,idx_obs], u[0,idx_obs]+u[1,idx_obs]+u[3,idx_obs], u[2,idx_obs]+u[4,idx_obs]

def log_likelihood(m,T_obs,V_obs,T_0,T,sigma_DT=200,sigma_DV=250,sim_info=None):
    dim = T_obs.shape[0]
    # C_DT_inv = np.zeros((dim,dim))
    # C_DT = np.zeros((dim, dim))
    # np.fill_diagonal(C_DT_inv, 1/sigma_DT)
    # np.fill_diagonal(C_DT, sigma_DT)

    C_DV_inv = np.zeros((dim, dim))
    # C_DV = np.zeros((dim, dim))
    np.fill_diagonal(C_DV_inv, 1/sigma_DV)
    # np.fill_diagonal(C_DV,sigma_DV)

    _, T_tot, V_tot = forward_model(m=m,sim_info=sim_info)

    # chi = np.matmul(np.matmul((T_obs - T_tot).transpose(),C_DT_inv),(T_obs - T_tot))\
    #       + np.matmul(np.matmul((V_obs - V_tot).transpose(),C_DV_inv),(V_obs - V_tot))
    # chi = np.matmul(np.matmul((np.log(T_obs) - np.log(T_tot)).transpose(), C_DT_inv), (np.log(T_obs) - np.log(T_tot))) \
    #       + np.matmul(np.matmul((np.log(V_obs) - np.log(V_tot)).transpose(), C_DV_inv), (np.log(V_obs) - np.log(V_tot)))
    chi = np.matmul(np.matmul((np.log(V_obs) - np.log(V_tot)).transpose(), C_DV_inv), (np.log(V_obs) - np.log(V_tot)))
    return -(T_0*chi)/(2.0*T) #- 2.0*np.log(2.0*np.pi) - 0.5*np.log(np.linalg.det(C_DT)*np.linalg.det(C_DV))

def d_obs(T_tot, V_tot, sigma_DT, sigma_DV):
    #TODO: check variance (root? exponential?)
    T_obs=np.abs(np.random.normal(loc=T_tot,scale=np.sqrt(sigma_DT),size=T_tot.shape))
    V_obs=np.abs(np.random.normal(loc=V_tot, scale=np.sqrt(sigma_DV), size=V_tot.shape))
    return T_obs, V_obs

def T_distribution(x,lb,p,ub):
    if lb <= x and x <= p:
        return (x-lb)**2/((ub-lb)*(p-lb))
    elif p < x and x  <= ub:
        return 1.0 - (ub-x)**2/((ub-lb)*(ub-p))
    elif x < ub:
        return 0
    elif x > p:
        return 1

def U_distribution(x,lb,ub):
    if lb <= x and x <= ub:
        # return (x-lb)/(ub-lb)
        return 1.0/(ub-lb)
    else:
        return 0
    # elif x > ub:
    #     return 1
    # else:
    #     return 0

def log_prior_model(m, T_0, T):
    ranges = model_space()
    priors = np.zeros((m.shape[0],))
    for i in range(0,m.shape[0]):
        priors[i] = U_distribution(m[i],ranges[i,0],ranges[i,1])
        if priors[i] == 0:
            return 0
    p_joint = np.sum(np.log(priors))
    assert p_joint <= 0
    return (T_0*p_joint)/T
    # p_lambda = U_distribution(m[0],ranges[0,0],ranges[0,1])
    # p_gamma = U_distribution(m[1],ranges[1,0],ranges[1,1])
    # p_ks = U_distribution(m[2],ranges[2,0],ranges[2,1])
    # p_kr = U_distribution(m[3],ranges[3,0],ranges[3,1])
    # p_delta = U_distribution(m[4],ranges[4,0],ranges[4,1])
    # p_Ns = U_distribution(m[5],ranges[5,0],ranges[5,1])
    # p_c = U_distribution(m[6],ranges[6,0],ranges[6,1])
    # p_Nr = U_distribution(m[7],ranges[7,0],ranges[7,1])


    # # p_lambda = T_distribution(m[0],np.log(1.595*1e4),np.log(3.19*1e4),np.log(5.104*1e4))
    # # p_lambda = T_distribution(m[0],np.log(1e3),np.log(1e4),np.log(1e5))
    # p_lambda = U_distribution(m[0],np.log(1e3),np.log(1e5))
    # # p_gamma = T_distribution(m[1], np.log(0.005),np.log(0.01),np.log(0.016))
    # p_gamma = U_distribution(m[1], np.log(0.005),np.log(0.016))
    # p_ks = U_distribution(m[2],np.log(1.2*1e-8),np.log(3.6*1e-8))
    # p_kr = U_distribution(m[3], np.log(1.0 * 1e-8),np.log(3.0 * 1e-8))
    # # p_delta = T_distribution(m[4], np.log(0.25),np.log(1.0),np.log(1.5))
    # p_delta = U_distribution(m[4], np.log(0.25),np.log(1.5))
    # p_Ns = U_distribution(m[5],np.log(2000),np.log(4000))
    # # p_c = T_distribution(m[6],np.log(9.1),np.log(23),np.log(36))
    # p_c = U_distribution(m[6],np.log(9.1),np.log(36))
    # p_Nr = U_distribution(m[7], np.log(1000),np.log(3000))
    # p_joint = np.log(p_lambda)+np.log(p_gamma)+np.log(p_ks)+np.log(p_kr)+np.log(p_delta)+np.log(p_Ns)+np.log(p_c)+np.log(p_Nr)
    # assert p_joint <= 0
    # return (T_0*p_joint)/T

def proposal(m_old=None,mode=None):
    # symmetric random walk Metropolis
    m_proposed = np.zeros((8,))
    ranges = model_space()
    if m_old is None and mode == 'init': # initialize m
        for i in range(0,8):
            m_proposed[i] = np.random.uniform(ranges[i,0],ranges[i,1])
        return m_proposed
    elif m_old is not None and mode=='uniform_ind':
        for i in range(0,8):
            m_proposed[i] = np.random.uniform(ranges[i, 0], ranges[i, 1])
        return m_proposed, 0
    elif m_old is not None and mode=='uniform':
        factor = 0.25
        for i in range(0,8):
            m_proposed[i] = m_old[i] + factor*np.random.uniform((ranges[i, 0]-m_old[i]), (ranges[i, 1]-m_old[i]))
        return m_proposed, 0
    elif m_old is not None and mode =='normal':
        # std = np.array(np.eye(m_old.shape[0])*[5/10, 1/100, 1/500, 1/500, 5/100, 2/10, 1/50, 2/10])
        std = np.array(np.eye(m_old.shape[0])*[1/100, 1/100, 1/100, 1/100, 1/100, 1/100, 1/100, 1/100])
        m_proposed = np.random.multivariate_normal(m_old, std**2)
        trial = 1
        while (m_proposed < ranges[:,0]).any() or (m_proposed > ranges[:,1]).any():
            m_proposed = np.random.multivariate_normal(m_old, (std/trial)**2)
            print('Correction of sigma -- new std = ',np.diagonal(std)/trial)

        # for i in range(0,8):
        #     m_proposed[i] = m_old[i] + np.random.normal(0,sigma[i])
        #     trial = 1
        #     while m_proposed[i] < ranges[i,0] or m_proposed[i] > ranges[i,1]:
        #         trial += 1
        #         m_proposed[i] = m_old[i] + np.random.normal(0,sigma[i]/trial)
        #         print('Correction of sigma for i = ',i,'-- new sigma = ',sigma[i]/trial)
        # C = np.zeros((m_old.shape[0],m_old.shape[0]))
        # np.fill_diagonal(C,(ranges[:,1]-ranges[:,0])/10)
        # m_proposed = np.random.multivariate_normal(m_old,C)
        return m_proposed, 0
    elif m_old is not None and mode =='trunc_normal':
        transition_prob_old_new = 0
        transition_prob_new_old = 0
        for i in range(0,8):
            std = (ranges[i,1]-ranges[i,0])/15
            m_proposed[i] = stats.truncnorm.rvs(ranges[i,0], ranges[i,1],loc=m_old[i], scale=std)
            transition_prob_old_new += np.log(stats.truncnorm.pdf(m_old[i],ranges[i,0], ranges[i,1],loc=m_old[i], scale=std))
            transition_prob_new_old += np.log(stats.truncnorm.pdf(m_proposed[i],ranges[i,0], ranges[i,1],loc=m_old[i], scale=std))
        return m_proposed, transition_prob_new_old-transition_prob_old_new
    elif m_old is not None and mode =='beta':
        transition_prob_old_new = 0
        transition_prob_new_old = 0
        for i in range(0,8):
            m_norm_old = (m_old[i]-ranges[i,0])/(ranges[i,1]-ranges[i,0])
            n = 0.1
            para1 = n*m_norm_old
            para2 = n*(1-m_norm_old)
            m_norm_new = stats.beta.rvs(para1, para2, loc=m_norm_old)
            m_proposed[i] = m_norm_new*(ranges[i,1]-ranges[i,0])+ranges[i,0]
            transition_prob_old_new += np.log(stats.beta.pdf(m_norm_old,para1,para2))
            transition_prob_new_old += np.log(stats.beta.pdf(m_norm_new,para1,para2))
        return m_proposed, transition_prob_new_old-transition_prob_old_new
    else:
        return 0

if __name__ == '__main__':
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    '''Defining simulation parameters'''
    tol = 1e-8
    number_of_samples = 10000
    samples = np.zeros((number_of_samples,8))
    likelihoods = np.zeros((number_of_samples, ))
    T_schedule = np.zeros((number_of_samples, ))
    t_0 = 0
    # t_end = 56
    t_end = 28
    # h = 0.001
    h = 0.1
    Nt = int(1 + (t_end - t_0) / h)
    t = np.linspace(t_0, t_end, Nt)
    assert t[-1] == t_end

    sim = dict()
    sim['Nt'] = Nt
    sim['h'] = h
    sim['sampling_rate'] = 2
    sigma_DT=1
    sigma_DV=250
    T_0 = 1.0
    T = T_0
    cooling_constant = 0.999
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    '''INIT: model and parameters'''
    m_true, _, _ = model_Rong_init()
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    '''FOWARD MODEL'''
    u_true, T_tot_true, V_tot_true = forward_model(m=m_true,sim_info=sim)
    T_obs, V_obs = d_obs(T_tot_true,V_tot_true,sigma_DT,sigma_DV)

    m_old = proposal(m_old=None,mode='init')
    samples[0,:] = m_old
    L_old = log_likelihood(m_old,T_obs,V_obs,T_0,T,sim_info=sim)
    m_prior_old = log_prior_model(m_old,T_0,T)
    likelihoods[0] = L_old
    T_schedule[0] = T
    acceptance_rate = 0

    ax0a, ax0b, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = plot_chains(likelihoods,
                                                                     T_schedule,
                                                                     samples, m_true,
                                                                     number_of_samples)

    for k in range(1,number_of_samples):
        m_tmp, transition_porb = proposal(m_old=samples[k-1,:],mode='normal')
        L_tmp = log_likelihood(m_tmp,T_obs,V_obs,T_0,T,sim_info=sim)
        # m_prior_tmp = log_prior_model(m_tmp,T_0,T)

        # alpha = min(1,np.exp(L_tmp+m_prior_tmp-L_old-m_prior_old + transition_porb))
        alpha = min(1,np.exp(L_tmp-L_old))
        u = np.random.uniform(0,1)
        if u < alpha:
            acceptance_rate += 1
            L_old = L_tmp
            samples[k,:] = m_tmp
        else:
            samples[k,:] = samples[k-1,:]
        likelihoods[k] = L_old
        if k%25 == 0:
            T = cooling_constant*T
        T_schedule[k] = T
        print('acceptance rate and Temp. in ',k, ': ', acceptance_rate/k, ', ',T)
        if k%250 == 0:
            print('Iteration ', k)
        if k%1000 == 0:
            ax0a.plot(range(k-1000,k), np.exp(likelihoods[k-1000:k]))
            ax0b.plot(range(k-1000,k), T_schedule[k-1000:k])
            ax1.plot(samples[k-1000:k, 0], range(k-1000,k))
            ax2.plot(samples[k-1000:k, 1], range(k-1000,k))
            ax3.plot(samples[k-1000:k, 2], range(k-1000,k))
            ax4.plot(samples[k-1000:k, 3], range(k-1000,k))
            ax5.plot(samples[k-1000:k, 4], range(k-1000,k))
            ax6.plot(samples[k-1000:k, 5], range(k-1000,k))
            ax7.plot(samples[k-1000:k, 6], range(k-1000,k))
            ax8.plot(samples[k-1000:k, 7], range(k-1000,k))

    figure = corner.corner(samples,
                           labels=[r"$\lambda$", r"$\gamma$", r"$k_s$", r"$k_r$", r"$\delta$", r"$N_s$", r"$c$", r"$N_r$"],
                           show_titles=True, title_kwargs={"fontsize": 12})

    print('Finish')