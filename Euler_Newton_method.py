import numba
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from model import f_Wu, Df_Wu, model_Wu_init
from model import f_Rong, Df_Rong, model_Rong_init
from matplotlib import rc
rc('font',**{'family':'sans-serif','serif':['Computer Modern Roman'],'sans-serif':['Helvetica']})
rc('text', usetex=True)
matplotlib.rcParams['text.usetex'] = True
'''Defining simulation parameters'''
tol = 1e-8
t_0 = 0
t_end = 800  # first two years
h = 0.001
Nt = int(1 + (t_end - t_0) / h)
t = np.linspace(t_0, t_end, Nt)
assert t[-1] == t_end

'''model and parameters'''
u = np.zeros((5,Nt))

# para, u[:,0] = model_Wu_init()
para, u[:,0] = model_Rong_init()

@numba.jit(nopython=True)
def g(u_old, u_new, h, f_new):
    return u_new - u_old - h * f_new

@numba.jit(nopython=True)
def Dg(h, Df):
    return np.identity(5) - h * Df

@numba.jit(nopython=True)
def Newton(x_old, u_old, p = para):
    J_old = Dg(h, Df_Rong(x_old, p=para))
    g_old = g(u_old, x_old, h, f_Rong(x_old, p=para))
    return x_old - np.dot(np.linalg.pinv(J_old), g_old)

def NDF2(x_old, u_old, x_init, h):
    kappa = -1/9
    gamma = 1.5
    A = np.identity(5) - 0.6*h*Df_Wu(x_old)
    invA = np.linalg.inv(A)
    psi = 3 / 2 * u_old[:, -1] - 12/5 * u_old[:, -2] - 9/10 * u_old[:, -3]
    rhs = 0.6*h*f_Wu(x_old) - psi - (x_old - x_init)
    x_new = x_old + np.matmul(invA, rhs)
    return x_new

solver = 'Euler'

if solver == 'Euler':
    for i in range(1, Nt):
        guess = u[:, i - 1] + h * f_Rong(u[:, i - 1])

        x_old = guess
        x_new = Newton(x_old, u[:, i-1])
        k = 0
        dev = np.abs(x_old - x_new)
        while dev[0] > tol and dev[1] > tol and dev[2] > tol:
            x_old = x_new
            x_new = Newton(x_old, u[:, i-1])
            dev = np.abs(x_old - x_new)
            k += 1
        u[:, i] = x_new
        if not (i % 10000):
            print('Solved at time ', str(i), ' after ', str(k), ' Newton iterations.')

elif solver == 'Wolfbrandt':
    for i in range(1, Nt):
        print(str(i))
        f1 = f_Wu(u[:, i - 1], p=para)
        Jf = Df_Wu(u[:, i - 1], p=para)
        d = 1 / (2 + np.sqrt(2))
        W = np.identity(5) - h * d * Jf
        W_inv = np.linalg.inv(W)
        k1 = np.matmul(W_inv, f1)

        f2 = f_Wu(u[:, i - 1] + 2 / 3 * h * k1, p=para)
        k2 = np.matmul(W_inv, f2 - 4 / 3 * h * d * np.matmul(Jf, k1))

        u[:, i] = u[:, i - 1] + 0.25 * h * (k1 + 3 * k2)

elif solver == 'NDF2':
    for i in range(1,3):
        u[:, i] = u[:, 0]
        # guess = u[:, i - 1] + h * f_Wu(u[:, i-1])
        #
        # x_old = guess
        # x_new = Newton(x_old, u[:, i - 1])
        # dev = np.abs(x_old - x_new)
        # while dev[0] > tol and dev[1] > tol and dev[2] > tol:
        #     x_old = x_new
        #     x_new = Newton(x_old, u[:, i - 1])
        #     dev = np.abs(x_old - x_new)
        # u[:, i] = x_new

    for i in range(3, Nt):
        print(str(i))
        if i == 100:
            print(str(i))
        guess = 3*u[:,i-1] - 3*u[:,i-2] - u[:,i-3]

        x_new = NDF2(guess,u[:,i-3:i],guess,h)
        # try:
        #     if x_new == 0:
        #         print(str(i))
        # except ValueError:
        #     pass
        # dev = np.abs(x_old - x_new)
        # while dev[0] > tol and dev[1] > tol and dev[2] > tol:
        for m in range(0,2):
            x_old = x_new
            x_new = NDF2(x_old,u[:,i-3:i],guess,h)
            # dev = np.abs(x_old - x_new)
        u[:, i] = x_new
        # if not (i % 100):
        #     print('Solved at time ', str(i), ' after ', str(k), ' Newton iterations.')




np.save('data/eRT_04_alpha_02/u.npy', u)
fig1 = plt.figure(1)
plt.plot(t, u[0,:]*10e-3, 'black', label = r'$\displaystyle uninfected \; T \; cells$')
plt.grid()
plt.xlim((0, 400))
plt.xlabel(r'$\displaystyle time \; [days]$', labelpad=10, fontsize = 18)
plt.ylabel(r'$\displaystyle cell \; concentration \; [cells/\mu l]$', labelpad=10, fontsize=18)
plt.tick_params(labelsize=16)
plt.tight_layout()
plt.show()
plt.savefig('images/eRT_0_alpha_0/treated_T.pdf', dpi = 300, pad_inches=0.25)
# ttl1 = plt.title(r'$\displaystyle uninfected \;\; T \;\; cells$', fontsize = 15)
# ttl1.set_position([.5, 1.25])

# plt.subplot(3,1,2)
fig2 = plt.figure(2)
plt.yscale('log')
plt.plot(t, u[1,:], 'black', label = 'Ts')
plt.grid()
plt.xlim((0, 400))
# plt.legend()
plt.xlabel(r'$\displaystyle time \; [days]$', labelpad=10, fontsize = 18)
plt.ylabel(r'$\displaystyle cell \; concentration \; [log_{10}(cells/ml^3)]$', labelpad=10, fontsize=18)
plt.show()
plt.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig('images/eRT_0_alpha_0/treated_Ts.pdf', dpi = 300, pad_inches=0.25)

fig3 = plt.figure(3)
plt.yscale('log')
plt.plot(t, u[3,:], 'black', label = 'Vs')
plt.grid()
plt.xlim((0, 400))
# plt.ylim((-10, 10))
plt.xlabel(r'$\displaystyle time \; [days]$', labelpad=10, fontsize = 18)
plt.ylabel(r'$\displaystyle cell \; concentration \; [log_{10}(cells/ml^3)]$', labelpad=10, fontsize=18)
plt.tick_params(labelsize=16)
plt.show()
plt.tight_layout()
plt.savefig('images/eRT_0_alpha_0/treated_Vs.pdf', dpi = 300, pad_inches=0.25)

# plt.subplot(3,1,3)
fig4 = plt.figure(4)
plt.yscale('log')
plt.plot(t, u[2,:], 'black', label = 'Tr')
plt.grid()
# plt.legend()
plt.xlabel(r'$\displaystyle time \; [days]$', labelpad=10, fontsize = 18)
plt.ylabel(r'$\displaystyle cell \; concentration \; [log_{10}(cells/ml^3)]$', labelpad=10, fontsize=18)
plt.show()
plt.tight_layout()
plt.tick_params(labelsize=16)
plt.savefig('images/eRT_0_alpha_0/treated_Tr.pdf', dpi = 300, pad_inches=0.25)

fig5 = plt.figure(5)
plt.yscale('log')
plt.plot(t, u[4,:], 'black', label = 'Vr')
plt.grid()
plt.xlim((0, 400))
# plt.ylim((-15, 5))
plt.xlabel(r'$\displaystyle time \; [days]$', labelpad=10, fontsize = 18)
plt.ylabel(r'$\displaystyle cell \; concentration \; [log_{10}(cells/ml^3)]$', labelpad=10, fontsize=18)
plt.tick_params(labelsize=16)
plt.show()
plt.tight_layout()
plt.savefig('images/eRT_0_alpha_0/treated_Vr.pdf', dpi = 300, pad_inches=0.25)

fig6 = plt.figure(6)
plt.yscale('log')
plt.plot(t, u[4,:] + u[3,:], 'black', label = 'Vr')
plt.grid()
plt.xlim((0, 400))
# plt.ylim((-10, 10))
plt.xlabel(r'$\displaystyle time \; [days]$', labelpad=10, fontsize = 18)
plt.ylabel(r'$\displaystyle cell \; concentration \; [log_{10}(cells/ml^3)]$', labelpad=10, fontsize=18)
plt.tick_params(labelsize=16)
plt.show()
plt.tight_layout()
plt.savefig('images/eRT_0_alpha_0/treated_total_V.pdf', dpi = 300, pad_inches=0.25)

fig7 = plt.figure(7)
plt.yscale('log')
plt.plot(t, u[4,:] + u[3,:], '0.65', label = '$\displaystyle V_r + V_s$')
plt.plot(t, u[3,:], 'black', Linestyle = ':', label = '$\displaystyle V_s$')
plt.plot(t, u[4,:], 'black', Linestyle = '--', label = '$\displaystyle V_r$')
plt.grid()
plt.legend()
plt.xlim((0, 400))
# plt.ylim((-10, 10))
plt.xlabel(r'$\displaystyle time \; [days]$', labelpad=10, fontsize = 18)
plt.ylabel(r'$\displaystyle cell \; concentration \; [log_{10}(cells/ml^3)]$', labelpad=10, fontsize=18)
plt.tick_params(labelsize=16)
plt.show()
plt.tight_layout()
plt.savefig('images/eRT_0_alpha_0/treated_overview_V.pdf', dpi = 300, pad_inches=0.25)

fig8 = plt.figure(8)
plt.yscale('log')
plt.plot(t, u[2,:] + u[1,:], '0.65', label = '$\displaystyle T_r + T_s$')
plt.plot(t, u[1,:], 'black', Linestyle = ':', label = '$\displaystyle T_s$')
plt.plot(t, u[2,:], 'black', Linestyle = '--', label = '$\displaystyle T_r$')
plt.grid()
plt.legend()
plt.xlim((0, 400))
# plt.ylim((-10, 10))
plt.xlabel(r'$\displaystyle time \; [days]$', labelpad=10, fontsize = 18)
plt.ylabel(r'$\displaystyle cell \; concentration \; [log_{10}(cells/ml^3)]$', labelpad=10, fontsize=18)
plt.tick_params(labelsize=16)
plt.show()
plt.tight_layout()
plt.savefig('images/eRT_0_alpha_0/treated_overview_infected_T.pdf', dpi = 300, pad_inches=0.25)

# plt.subplots_adjust(wspace = 0.4)
# plt.show()
#
# fig2 = plt.figure(2)
# plt.subplot(2,2,1)
# plt.plot(t, u[0,:], 'green', label = 'T')
# plt.grid()
# plt.legend()
# # plt.xlabel('time (days)')
# # plt.ylabel('cell count (cells/mm^3)')
#
# # plt.subplot(2,2,2)
# # plt.yscale('log')
# # plt.plot(t, u[1,:], 'yellow', label = 'Ts (wild Type)')
# # plt.grid()
# # plt.legend()
# # # plt.xlabel('time (days)')
# # plt.ylabel('cell count (cells/mm^3)')
#
# plt.subplot(2,2,2)
# plt.yscale('log')
# plt.plot(t, u[3,:], 'yellow', Linestyle = '--', label = 'Vs (wild type)')
# plt.grid()
# plt.legend()
# # plt.xlabel('time (days)')
# plt.ylabel('cell count (cells/mm^3)')
#
# # plt.subplot(2,2,3)
# # plt.yscale('log')
# # plt.plot(t, u[2,:], 'red', label = 'Tr')
# # plt.grid()
# # plt.legend()
# # plt.xlabel('time (days)')
#
# plt.subplot(2,2,3)
# plt.yscale('log')
# plt.plot(t, u[4,:], 'red', Linestyle = '--', label = 'Vr (resistant)')
# plt.grid()
# plt.legend()
# plt.xlabel('time (days)')
# # plt.ylabel('cell count (cells/mm^3)')
#
# plt.subplot(2,2,4)
# plt.yscale('log')
# plt.plot(t, u[4,:] + u[3,:], 'red', Linestyle = '--', label = 'Vr + Vs')
# plt.grid()
# plt.legend()
# plt.xlabel('time (days)')
#
# plt.subplots_adjust(wspace = 0.4)
# plt.show()

print('Finish')