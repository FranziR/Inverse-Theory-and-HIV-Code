import numba
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from model import f_Wu, Df_Wu, model_Wu_init
from model import f_Rong, Df_Rong, model_Rong_init
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'serif': ['Computer Modern Roman'], 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
matplotlib.rcParams['text.usetex'] = True
'''Defining simulation parameters'''
tol = 1e-8

t_0 = 0
t_end = 800  # first two days
h = 0.001
Nt = int(1 + (t_end - t_0) / h)
t = np.linspace(t_0, t_end, Nt)
assert t[-1] == t_end

'''model and parameters'''
steady_states = np.zeros((11, 5))
u = np.zeros((5, Nt))
epsilon_RT_alpha_02 = np.linspace(0, 1, 11)


@numba.jit(nopython=True)
def g(u_old, u_new, h, f_new):
    return u_new - u_old - h * f_new


@numba.jit(nopython=True)
def Dg(h, Df):
    return np.identity(5) - h * Df


for idx, eps in enumerate(epsilon_RT_alpha_02):
    print('Solving for epsilon_RT = ', str(eps))
    para, u[:, 0] = model_Rong_init()
    para[8] = eps


    @numba.jit(nopython=True)
    def Newton(x_old, u_old, p=para):
        J_old = Dg(h, Df_Rong(x_old, p=para))
        g_old = g(u_old, x_old, h, f_Rong(x_old, p=para))
        return x_old - np.dot(np.linalg.pinv(J_old), g_old)


    for i in range(1, Nt):
        guess = u[:, i - 1] + h * f_Rong(u[:, i - 1])

        x_old = guess
        x_new = Newton(x_old, u[:, i - 1], p = para)
        k = 0
        dev = np.abs(x_old - x_new)
        while dev[0] > tol and dev[1] > tol and dev[2] > tol:
            x_old = x_new
            x_new = Newton(x_old, u[:, i - 1], p = para)
            dev = np.abs(x_old - x_new)
            k += 1
        u[:, i] = x_new
        if not (i % 10000):
            print('Solved at time ', str(i), ' after ', str(k), ' Newton iterations.')


    steady_states[idx, :] = u[:, -1]

np.save('data/evolution_over_epsilon/steady_states.npy', steady_states)
fig1 = plt.figure(1)
plt.plot(t, u[0, :] * 10e-3, 'black', label=r'$\displaystyle uninfected \; T \; cells$')
plt.grid()
plt.xlim((0, 400))
plt.xlabel(r'$\displaystyle time \; [days]$', labelpad=10, fontsize=12)
plt.ylabel(r'$\displaystyle cell \; concentration \; [cells/\mu l]$', labelpad=10, fontsize=12)
plt.tight_layout()
plt.show()
plt.savefig('images/eRT_08_alpha_05/treated_T.pdf', dpi=300, pad_inches=0.25)

fig2 = plt.figure(2)
plt.yscale('log')
plt.plot(t, u[1, :], 'black', label='Ts')
plt.grid()
plt.xlim((0, 400))
plt.xlabel(r'$\displaystyle time \; [days]$', labelpad=10, fontsize=12)
plt.ylabel(r'$\displaystyle cell \; concentration \; [log_{10}(cells/ml^3)]$', labelpad=10, fontsize=12)
plt.show()
plt.tight_layout()
plt.savefig('images/eRT_08_alpha_05/treated_Ts.pdf', dpi=300, pad_inches=0.25)

fig3 = plt.figure(3)
plt.yscale('log')
plt.plot(t, u[3, :], 'black', label='Vs')
plt.grid()
plt.xlim((0, 400))
plt.xlabel(r'$\displaystyle time \; [days]$', labelpad=10, fontsize=12)
plt.ylabel(r'$\displaystyle cell \; concentration \; [log_{10}(cells/ml^3)]$', labelpad=10, fontsize=12)
plt.show()
plt.tight_layout()
plt.savefig('images/eRT_08_alpha_05/treated_Vs.pdf', dpi=300, pad_inches=0.25)

fig4 = plt.figure(4)
plt.yscale('log')
plt.plot(t, u[2, :], 'black', label='Tr')
plt.grid()
plt.xlabel(r'$\displaystyle time \; [days]$', labelpad=10, fontsize=12)
plt.ylabel(r'$\displaystyle cell \; concentration \; [log_{10}(cells/ml^3)]$', labelpad=10, fontsize=12)
plt.show()
plt.tight_layout()
plt.savefig('images/eRT_08_alpha_05/treated_Tr.pdf', dpi=300, pad_inches=0.25)

fig5 = plt.figure(5)
plt.yscale('log')
plt.plot(t, u[4, :], 'black', label='Vr')
plt.grid()
plt.xlim((0, 400))
plt.xlabel(r'$\displaystyle time \; [days]$', labelpad=10, fontsize=12)
plt.ylabel(r'$\displaystyle cell \; concentration \; [log_{10}(cells/ml^3)]$', labelpad=10, fontsize=12)
plt.show()
plt.tight_layout()
plt.savefig('images/eRT_08_alpha_05/treated_Vr.pdf', dpi=300, pad_inches=0.25)

fig6 = plt.figure(6)
plt.yscale('log')
plt.plot(t, u[4, :] + u[3, :], 'black', label='Vr')
plt.grid()
plt.xlim((0, 400))
plt.xlabel(r'$\displaystyle time \; [days]$', labelpad=10, fontsize=12)
plt.ylabel(r'$\displaystyle cell \; concentration \; [log_{10}(cells/ml^3)]$', labelpad=10, fontsize=12)
plt.show()
plt.tight_layout()
plt.savefig('images/eRT_08_alpha_05/treated_total_V.pdf', dpi=300, pad_inches=0.25)

print('Finish')
