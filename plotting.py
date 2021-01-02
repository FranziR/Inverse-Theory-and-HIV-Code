import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'serif': ['Computer Modern Roman'], 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
matplotlib.rcParams['text.usetex'] = True
import os

file_path = os.path.dirname(os.path.abspath(__file__))
t_0 = 0
t_end = 800  # first two days
h = 0.001
Nt = int(1 + (t_end - t_0) / h)
t = np.linspace(t_0, t_end, Nt)

folder = 'eRT_04_alpha_02'
u = np.load(os.path.join(file_path, 'data', folder, 'u.npy'))
fontsize_labels = 15
label_pad_x = 5
label_pad_y = 10

fig1 = plt.figure(1)
plt.plot(t, u[0, :] * 10e-3, 'black', label=r'$\displaystyle uninfected \; T \; cells$')
plt.grid()
plt.xlim((0, t_end))
plt.xlabel(r'$\displaystyle time \; [days]$', labelpad=label_pad_y, fontsize=fontsize_labels)
plt.ylabel(r'$\displaystyle cell \; concentration \; [cells/\mu l]$', labelpad=label_pad_x, fontsize=fontsize_labels)
plt.tick_params(labelsize=16)
plt.tight_layout()
plt.show()
# plt.savefig(os.path.join('images', folder, 'treated_T.pdf'), dpi=300, pad_inches=0.25)

fig2 = plt.figure(2)
plt.yscale('log')
plt.plot(t, u[1, :], 'black', label='Ts')
plt.grid()
plt.xlim((0, t_end))
plt.xlabel(r'$\displaystyle time \; [days]$', labelpad=label_pad_y, fontsize=fontsize_labels)
plt.ylabel(r'$\displaystyle cell \; concentration \; [log_{10}(cells/ml^3)]$', labelpad=label_pad_x, fontsize=fontsize_labels)
plt.show()
plt.tick_params(labelsize=16)
plt.tight_layout()
# plt.savefig(os.path.join('images', folder, 'treated_Ts.pdf'), dpi=300, pad_inches=0.25)

fig3 = plt.figure(3)
plt.yscale('log')
plt.plot(t, u[3, :], 'black', label='Vs')
plt.grid()
plt.xlim((0, t_end))
plt.xlabel(r'$\displaystyle time \; [days]$', labelpad=label_pad_y, fontsize=fontsize_labels)
plt.ylabel(r'$\displaystyle cell \; concentration \; [log_{10}(cells/ml^3)]$', labelpad=label_pad_x, fontsize=fontsize_labels)
plt.tick_params(labelsize=16)
plt.show()
plt.tight_layout()
# plt.savefig(os.path.join('images', folder, 'treated_Vs.pdf'), dpi=300, pad_inches=0.25)

# plt.subplot(3,1,3)
fig4 = plt.figure(4)
plt.yscale('log')
plt.plot(t, u[2, :], 'black', label='Tr')
plt.grid()
# plt.legend()
plt.xlabel(r'$\displaystyle time \; [days]$', labelpad=label_pad_y, fontsize=fontsize_labels)
plt.ylabel(r'$\displaystyle cell \; concentration \; [log_{10}(cells/ml^3)]$', labelpad=label_pad_x, fontsize=fontsize_labels)
plt.show()
plt.tight_layout()
plt.tick_params(labelsize=16)
# plt.savefig(os.path.join('images', folder, 'treated_Tr.pdf'), dpi=300, pad_inches=0.25)

fig5 = plt.figure(5)
plt.yscale('log')
plt.plot(t, u[4, :], 'black', label='Vr')
plt.grid()
plt.xlim((0, t_end))
# plt.ylim((-15, 5))
plt.xlabel(r'$\displaystyle time \; [days]$', labelpad=label_pad_y, fontsize=fontsize_labels)
plt.ylabel(r'$\displaystyle cell \; concentration \; [log_{10}(cells/ml^3)]$', labelpad=label_pad_x, fontsize=fontsize_labels)
plt.tick_params(labelsize=16)
plt.show()
plt.tight_layout()
# plt.savefig(os.path.join('images', folder, 'treated_Vr.pdf'), dpi=300, pad_inches=0.25)

fig6 = plt.figure(6)
plt.yscale('log')
plt.plot(t, u[4, :] + u[3, :], 'black', label='Vr')
plt.grid()
plt.xlim((0, t_end))
# plt.ylim((-10, 10))
plt.xlabel(r'$\displaystyle time \; [days]$', labelpad=label_pad_y, fontsize=fontsize_labels)
plt.ylabel(r'$\displaystyle cell \; concentration \; [log_{10}(cells/ml^3)]$', labelpad=label_pad_x, fontsize=fontsize_labels)
plt.tick_params(labelsize=16)
plt.show()
plt.tight_layout()
# plt.savefig(os.path.join('images', folder, 'treated_total_V.pdf'), dpi=300, pad_inches=0.25)

fig7 = plt.figure(7)
plt.yscale('log')
plt.plot(t, u[4, :] + u[3, :], '0.65', Linewidth = 2, label='$\displaystyle V_r + V_s$')
plt.plot(t, u[3, :], 'black', Linestyle=':', label='$\displaystyle V_s$')
plt.plot(t, u[4, :], 'black', Linestyle='--', label='$\displaystyle V_r$')
plt.grid()
plt.legend(ncol = 3, fontsize = 16)
plt.xlim((0, t_end))
plt.xlabel(r'$\displaystyle time \; [days]$', labelpad=label_pad_y, fontsize=fontsize_labels)
plt.ylabel(r'$\displaystyle cell \; concentration \; [log_{10}(cells/ml^3)]$', labelpad=label_pad_x, fontsize=fontsize_labels)
plt.tick_params(labelsize=16)
plt.show()
plt.tight_layout()
# plt.savefig(os.path.join('images', folder, 'treated_overview_V.pdf'), dpi=300, pad_inches=0.25)

fig8 = plt.figure(8)
plt.yscale('log')
plt.plot(t, u[2,:] + u[1,:], '0.65', Linewidth = 2, label = '$\displaystyle T_r + T_s$')
plt.plot(t, u[1,:], 'black', Linestyle = ':', label = '$\displaystyle T_s$')
plt.plot(t, u[2,:], 'black', Linestyle = '--', label = '$\displaystyle T_r$')
plt.grid()
plt.legend(ncol = 3, fontsize = 16)
plt.xlim((0, t_end))
# plt.ylim((-10, 10))
plt.xlabel(r'$\displaystyle time \; [days]$', labelpad=label_pad_y, fontsize = fontsize_labels)
plt.ylabel(r'$\displaystyle cell \; concentration \; [log_{10}(cells/ml^3)]$', labelpad=label_pad_x, fontsize=fontsize_labels)
plt.tick_params(labelsize=16)
plt.show()
plt.tight_layout()
# plt.savefig(os.path.join('images', folder, 'treated_overview_infected_T.pdf'), dpi=300, pad_inches=0.25)

print('Finish')