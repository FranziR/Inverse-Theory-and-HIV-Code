import numpy as np
import numba

def model_Wu_init():
    '''Defining model parameters, taken from Wu et al.; temporal units in day!!'''
    para = dict()
    para['g'] = 10
    para['m1'] = 0.01
    para['m2'] = 0.02
    para['m3'] = 0.004
    para['p1'] = 5
    para['p2'] = 8
    para['c1'] = 30
    para['c2'] = 30
    para['c3'] = 3e-4
    para['c4'] = 5e-5
    para['uRT'] = 0
    para['uPI'] = 0
    para['uV'] = 0.2

    '''Defining initial values, following Kabanikhin et al.'''
    u = np.zeros((5,))

    u[0] = 900  # T
    u[1] = 70  # Ts
    u[2] = 30  # Tr
    u[3] = 500  # Vs
    u[4] = 0  # Vr

    return (para, u)

@numba.jit(nopython=True)
def model_Rong_init():
    '''Defining model parameters, taken from Wu et al.; temporal units in day!!'''
    p = np.zeros((14,))
    p[0] = 1e4 # lambda
    p[1] = 0.01  # d
    p[2] = 2.4e-8 # ks
    p[3] = 2e-8 # kr
    p[4] = 1 # delta
    p[5] = 3000 # Ns
    p[6] = 23 # c
    p[7] = 2000 # Nr
    p[8] = 0.4 # uRTs
    p[9] = 0 # uPIs
    p[10] = 0.2 # alpha
    p[11] = p[10] * p[8] # uRTr
    p[12] = p[10] * p[9] # uPIr
    p[13] = 3e-5 # uV

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

    return (p, u)

@numba.jit(nopython=True)
def f_Rong(u, p=None):
    if p is None:
        p, _ = model_Rong_init()

    lamb = p[0] # lambda
    d = p[1] # d
    ks = p[2]  # ks
    kr = p[3]  # kr
    delta = p[4]  # delta
    Ns = p[5]  # Ns
    c = p[6]  # c
    Nr = p[7]  # Nr
    uRTs = p[8]  # uRTs
    uPIs = p[9]  # uPIs
    alpha = p[10]  # alpha
    uRTr = p[11]  # uRTr
    uPIr = p[12] # uPIr
    uV = p[13]  # uV

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

def f_Wu(u, p=None):
    p, _ = model_Wu_init()
    T = u[0]
    Ts = u[1]
    Tr = u[2]
    Vs = u[3]
    Vr = u[4]

    return np.array([p['g'] - p['m1'] * T - p['c3'] * (1 - p['uRT']) * Vs * T - p['c4']*Vr*T,
                     p['c3'] * (1 - p['uRT']) * Vs * T - p['m2'] * Ts,
                     p['c4'] * Vr * T - p['m3'] * Tr,
                     p['p1'] * (1 - p['uV'])* (1 - p['uPI']) * Ts - p['c1'] * Vs,
                     p['p2'] * Tr + p['p1'] * p['uV'] * Ts - p['c2'] * Vr])

def Df_Wu(u, p=None):
    p, _ = model_Wu_init()

    T = u[0]
    Ts = u[1]
    Tr = u[2]
    Vs = u[3]
    Vr = u[4]

    Jf = np.zeros((5, 5))
    Jf[0, 0] = - p['m1'] - p['c3'] * (1 - p['uRT']) * Vs - p['c4'] * Vr
    Jf[0, 1] = 0
    Jf[0, 2] = 0
    Jf[0, 3] = - p['c3'] * (1 - p['uRT']) * T
    Jf[0, 4] = - p['c4'] * T

    Jf[1, 0] = p['c3'] * (1 - p['uRT']) * Vs
    Jf[1, 1] = - p['m2']
    Jf[1, 2] = 0
    Jf[1, 3] = p['c3'] * (1 - p['uRT']) * T
    Jf[1, 4] = 0

    Jf[2, 0] = p['c4'] * Vr
    Jf[2, 1] = 0
    Jf[2, 2] = -p['m3']
    Jf[2,3] = 0
    Jf[2,4] = p['c4']*T

    Jf[3, 0] = 0
    Jf[3, 1] = (1 - p['uPI']) * (1 - p['uV']) * p['p1']
    Jf[3, 2] = 0
    Jf[3, 3] = -p['c1']
    Jf[3, 4] = 0

    Jf[4, 0] = 0
    Jf[4, 1] = p['p1'] * p['uV']
    Jf[4, 2] = p['p2']
    Jf[4, 3] = 0
    Jf[4, 4] = -p['c2']

    return Jf

@numba.jit(nopython=True)
def Df_Rong(u, p=None):
    if p is None:
        p, _ = model_Rong_init()

    lamb = p[0]  # lambda
    d = p[1]  # d
    ks = p[2]  # ks
    kr = p[3]  # kr
    delta = p[4]  # delta
    Ns = p[5]  # Ns
    c = p[6]  # c
    Nr = p[7]  # Nr
    uRTs = p[8]  # uRTs
    uPIs = p[9]  # uPIs
    alpha = p[10]  # alpha
    uRTr = p[11]  # uRTr
    uPIr = p[12]  # uPIr
    uV = p[13]  # uV

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

def model_Adam_init():
    '''Defining model parameters, taken from Wu et al.; temporal units in day!!'''
    para = dict()
    para['lambda1'] = 1e5
    para['d1'] = 0.01
    para['k1'] = 8e-7
    para['lambda2'] = 31.98
    para['d2'] = 0.01
    para['k2'] = 1e-4
    para['delta'] = 0.7
    para['m1'] = 1e-5
    para['m2'] = 1e-5
    para['NT'] = 100
    para['c'] = 13
    para['rho1'] = 1
    para['rho2'] = 1
    para['lambdaE'] = 1
    para['bE'] = 0.3
    para['Kb'] = 100
    para['dE'] = 0.25
    para['Kd'] = 500
    para['deltaE'] = 0.1

    para['m3'] = 0.004
    para['p1'] = 5
    para['p2'] = 8
    para['c1'] = 30
    para['c2'] = 30
    para['c3'] = 3e-4
    para['c4'] = 5e-5
    para['uRT'] = 0
    para['uPI'] = 0
    para['uV'] = 0.2

    '''Defining initial values, following Kabanikhin et al.'''
    u = np.zeros((5,))

    u[0] = 900  # T
    u[1] = 70  # Ts
    u[2] = 30  # Tr
    u[3] = 500  # Vs
    u[4] = 0  # Vr

    return (para, u)