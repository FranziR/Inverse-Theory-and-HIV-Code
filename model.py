import numpy as np
import numba
import math

@numba.jit(nopython=True)
def model_Rong_init():
    '''Defining model parameters, taken from Wu et al.; temporal units in day!!'''
    p = np.zeros((8,))
    drugs = np.zeros((6,))
    p[0] = np.log(1e4) # lambda
    p[1] = np.log(0.01)  # d
    p[2] = np.log(2.4e-8) # ks
    p[3] = np.log(2e-8) # kr
    p[4] = np.log(1) # delta
    p[5] = np.log(3000) # Ns
    p[6] = np.log(23) # c
    p[7] = np.log(2000) # Nr

    drugs[0] = 0.4 # uRTs
    drugs[1] = 0 # uPIs
    drugs[2] = 0.2 # alpha
    drugs[3] = drugs[2] * drugs[0] # uRTr
    drugs[4] = drugs[2] * drugs[1] # uPIr
    drugs[5] = 3e-5 # uV

    # para = dict()
    # para['lambda'] = 1e4
    # para['d'] = 0.01
    # para['ks'] = 2.4e-8
    # para['kr'] = 2e-8
    # para['delta'] = 1
    # para['Ns'] = 3000
    # para['c'] = 23
    # para['Nr'] = 2000
    # para['uRTs'] = 0.4
    # para['uPIs'] = 0
    # para['alpha'] = 0.2
    # para['uRTr'] = para['alpha'] * para['uRTs']
    # para['uPIr'] = para['alpha'] * para['uPIs']
    # para['uV'] = 3e-5

    '''Defining initial values, following Kabanikhin et al.'''
    u = np.zeros((5,))

    u[0] = 3.19e5  # T
    u[1] = 6.81e3  # Ts
    u[2] = 0.46  # Tr
    u[3] = 8.88e5  # Vs
    u[4] = 39.95  # Vr

    # u[0] = 1e6  # T
    # u[1] = 0  # Ts
    # u[2] = 0  # Tr
    # u[3] = 1e-6  # Vs
    # u[4] = 0  # Vr

    return (p, drugs, u)

def model_space():
    m = np.zeros((8,2))

    m[0,0] = np.log(0.5e4) # lambda
    m[0,1] = np.log(1.5e4)
    m[1,0] = np.log(0.004)  # d
    m[1,1] = np.log(0.016)
    m[2,0] = np.log(1.2*1e-8) # ks
    m[2,1] = np.log(3.6*1e-8)
    m[3,0] = np.log(1.0 * 1e-8) # kr
    m[3,1] = np.log(3.0 * 1e-8)
    m[4,0] = np.log(0.75) # delta
    m[4,1] = np.log(1.5)
    m[5,0] = np.log(2000) # Ns
    m[5,1] = np.log(4000)
    m[6,0] = np.log(10) # c
    m[6,1] = np.log(36)
    m[7,0] = np.log(1000) # Nr
    m[7,1] = np.log(3000)
    return m

@numba.jit(nopython=True)
def f_Rong(u, p=None, drugs =None):
    if p is None:
        p, drugs, _ = model_Rong_init()

    lamb = p[0] # lambda
    d = p[1] # d
    ks = p[2]  # ks
    kr = p[3]  # kr
    delta = p[4]  # delta
    Ns = p[5]  # Ns
    c = p[6]  # c
    Nr = p[7]  # Nr

    uRTs = drugs[0]  # uRTs
    uPIs = drugs[1]  # uPIs
    alpha = drugs[2]  # alpha
    uRTr = drugs[3]  # uRTr
    uPIr = drugs[4] # uPIr
    uV = drugs[5]  # uV

    T = u[0]
    Ts = u[1]
    Tr = u[2]
    Vs = u[3]
    Vr = u[4]

    return np.array([lamb - d * T - ks * (1 - uRTs) * Vs * T - kr * (1 - uRTr) * Vr * T,
                     (1 - uV) * (1 - uRTs) * ks * Vs * T - delta * Ts,
                     uV * ks * (1 - uRTs) * Vs * T + kr * (1 - uRTr) * Vr * T - delta * Tr,
                     (1 - uPIs) * Ns * delta * Ts - c * Vs,
                     (1 - uPIr) * Nr * delta * Tr - c * Vr])

    # return np.array([p['lambda'] - p['d'] * T - p['ks'] * (1 - p['uRTs']) * Vs * T - p['kr']*(1 - p['uRTr']) *Vr*T,
    #                  (1 - p['uV']) *(1 - p['uRTs']) * p['ks'] * Vs * T - p['delta'] * Ts,
    #                  p['uV'] * p['ks'] * (1 - p['uRTs']) * Vs * T + p['kr'] *(1 - p['uRTr']) * Vr * T - p['delta'] * Tr,
    #                  (1 - p['uPIs']) * p['Ns'] * p['delta'] * Ts - p['c'] * Vs,
    #                  (1 - p['uPIr']) * p['Nr'] * p['delta'] * Tr - p['c'] * Vr])


@numba.jit(nopython=True)
def Df_Rong(u, p=None, drugs = None):
    if p is None and drugs is None:
        p, drugs, _ = model_Rong_init()

    lamb = p[0]  # lambda
    d = p[1]  # d
    ks = p[2]  # ks
    kr = p[3]  # kr
    delta = p[4]  # delta
    Ns = p[5]  # Ns
    c = p[6]  # c
    Nr = p[7]  # Nr

    uRTs = drugs[0]  # uRTs
    uPIs = drugs[1]  # uPIs
    alpha = drugs[2]  # alpha
    uRTr = drugs[3]  # uRTr
    uPIr = drugs[4]  # uPIr
    uV = drugs[5]  # uV

    T = u[0]
    Ts = u[1]
    Tr = u[2]
    Vs = u[3]
    Vr = u[4]

    Jf = np.zeros((5, 5))

    Jf[0, 0] = - d - ks * (1 - uRTs) * Vs - kr * (1 - uRTr) * Vr
    Jf[0, 1] = 0
    Jf[0, 2] = 0
    Jf[0, 3] = - ks * (1 - uRTs) * T
    Jf[0, 4] = - kr * (1 - uRTr) * T

    Jf[1, 0] = ks * (1 - uRTs) * (1 - uV) * Vs
    Jf[1, 1] = - delta
    Jf[1, 2] = 0
    Jf[1, 3] = ks * (1 - uRTs) * (1 - uV) * T
    Jf[1, 4] = 0

    Jf[2, 0] = uV*ks * (1 - uRTs) * Vs + kr * (1 - uRTr) * Vr
    Jf[2, 1] = 0
    Jf[2, 2] = -delta
    Jf[2, 3] = uV * ks * (1 - uRTs) * T
    Jf[2, 4] = kr * (1 - uRTr) * T

    Jf[3, 0] = 0
    Jf[3, 1] = (1 - uPIs) * Ns * delta
    Jf[3, 2] = 0
    Jf[3, 3] = -c
    Jf[3, 4] = 0

    Jf[4, 0] = 0
    Jf[4, 1] = 0
    Jf[4, 2] = Nr*delta * (1 - uPIr)
    Jf[4, 3] = 0
    Jf[4, 4] = -c

    return Jf

@numba.jit(nopython=True)
def g(u_old, u_new, h, f_new):
    return u_new - u_old - h * f_new

@numba.jit(nopython=True)
def Dg(h, Df):
    return np.identity(5) - h * Df

@numba.jit(nopython=True)
def Newton(x_old,u_old,p,drugs,h):
    J_old = Dg(h, Df_Rong(x_old,p=p,drugs=drugs))
    g_old = g(u_old, x_old, h, f_Rong(x_old, p=p, drugs=drugs))
    return x_old - np.dot(np.linalg.pinv(J_old), g_old)