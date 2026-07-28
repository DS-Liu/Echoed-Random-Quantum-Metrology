from functools import partial
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
from mpi4py.futures import MPIPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FuncFormatter

from fig2_statistics import forward, Gq_Gc_thetab_QFI_CFI, compute_Gc_vs_theta0, model
from utils import timing, mpl_rcParams, figwidth_dc, trim_zeros
mpl.rcParams.update(mpl_rcParams)

Path('data/scan_epsilon__TwoPhotonDriving').mkdir(exist_ok=True)
Path('figures').mkdir(exist_ok=True)
#------------------------------------------------------------------------------------------------------------------------------------#

@timing
def scan_epsilon():
    dim = 1050
    tau = .1 # $\tau/\chi=.1$
    N = 30 # number of time steps
    epsilon = np.logspace(0, 2, 21)[:19] # $\epsilon/\chi \in [1, 63.1]$
    n_samples = 1000

    rng = np.random.default_rng(2025)
    driving_list = np.zeros((len(epsilon), n_samples, 2, N))
    for i in range(len(epsilon)):
        driving_list[i] = rng.uniform(-epsilon[i], epsilon[i], size=(n_samples, 2, N))
    np.save('data/scan_epsilon__TwoPhotonDriving/driving_list.npy', driving_list)

    with MPIPoolExecutor() as executor:
        for i in tqdm(range(len(epsilon)), desc='forward'):
            n_ev = np.zeros((n_samples, N))
            psi0 = np.empty((n_samples, N), dtype=object)            
            result = executor.map(partial(forward, dim, tau, 'TwoPhotonDriving'), driving_list[i])
            for j, res in enumerate(result):
                n_ev[j], psi0[j] = res

            np.save(f'data/scan_epsilon__TwoPhotonDriving/n_ev__epsilon{i}.npy', n_ev)
            np.save(f'data/scan_epsilon__TwoPhotonDriving/psi0__epsilon{i}.npy', psi0)


        result = np.empty((len(epsilon), n_samples, N, 5))
        for i in tqdm(range(len(epsilon)), desc='Gq_Gc_thetab_QFI_CFI'):
            psi0 = np.load(f'data/scan_epsilon__TwoPhotonDriving/psi0__epsilon{i}.npy', allow_pickle=True) # psi0 here is of shape (n_samples, N) with dtype=object.
            temp = list(executor.map(Gq_Gc_thetab_QFI_CFI, psi0.reshape(-1)))
            result[i] = np.array(temp).reshape((n_samples, N, 5))
        np.save('data/scan_epsilon__TwoPhotonDriving/Gq_Gc_thetab_QFI_CFI.npy', result)

@timing
def plot_figS6():
    tau = .1
    epsilon = np.logspace(0, 2, 21)[:19]
    n_samples = 1000

    fig, axs = plt.subplots(2, 3, layout='constrained', figsize=(figwidth_dc, figwidth_dc*.58))

    epsilon_index = [13, 15] # corresponds to $\epsilon/\chi = [20, 31.6]$.
    N_index = 7 # corresponds to $\chi t=0.8$.
    data = [np.load(f'data/scan_epsilon__TwoPhotonDriving/psi0__epsilon{i}.npy', allow_pickle=True)[:, N_index] for i in epsilon_index] # psi0 here is of shape (2, n_samples).
    theta0 = np.linspace(0, .1, 101)
    Gc_theta0 = np.array([[compute_Gc_vs_theta0(psi0_, theta0) for psi0_ in psi0] for psi0 in data]) # Gc_theta0 here is of shape (2, n_samples, len(theta0))
    Gc_theta0_ave = [np.mean(A, axis=0) for A in Gc_theta0]
    Gc_theta0_std = [np.std(A, axis=0) for A in Gc_theta0]
    color = ['lightgreen', 'pink']
    for i in range(len(epsilon_index)):
        axs[0, 0].plot(theta0, Gc_theta0_ave[i], color=color[i])
        axs[0, 0].fill_between(theta0, Gc_theta0_ave[i]-Gc_theta0_std[i], Gc_theta0_ave[i]+Gc_theta0_std[i], color=color[i], edgecolor='none', alpha=.7)
    axs[0, 0].set_xlim(-1e-3, theta0.max()+1e-3)
    axs[0, 0].set_ylim(0, np.max(Gc_theta0_ave[1]+Gc_theta0_std[1]+1))
    axs[0, 0].minorticks_on()
    axs[0, 0].xaxis.set_major_locator(MultipleLocator(0.05))
    axs[0, 0].yaxis.set_major_locator(MultipleLocator(10))
    axs[0, 0].set_xlabel(r'$\theta_0$')
    axs[0, 0].set_ylabel(r'$G_\mathrm{c}(\theta_0)$')
    axs[0, 0].xaxis.set_major_formatter(FuncFormatter(trim_zeros))

    result = np.load('data/scan_epsilon__TwoPhotonDriving/Gq_Gc_thetab_QFI_CFI.npy') # result is of shape (len(epsilon), n_samples, N, 5).
    sample0 = result[epsilon_index[0], :, N_index, 1]
    sample1 = result[epsilon_index[1], :, N_index, 1]
    ax_inset = axs[0, 0].inset_axes([.44, .44, .52, .52])
    counts0, _, _ = ax_inset.hist(sample0, bins=np.arange(101)-.5, width=.5, color=color[0], align='mid') # Gc for $\epsilon/\chi=100, \chi t=2$.
    ax_inset.plot(np.arange(101), norm.pdf(np.arange(101), sample0.mean()-.25, sample0.std())*n_samples, color=color[0], linewidth=.45, alpha=.8)
    counts1, _, _ = ax_inset.hist(sample1, bins=np.arange(101)-.5, width=.5, color=color[1], align='right') # Gc for $\epsilon/\chi=200, \chi t=2$.
    ax_inset.plot(np.arange(101), norm.pdf(np.arange(101), sample1.mean()+.25, sample1.std())*n_samples, color=color[1], linewidth=.45, alpha=.8)
    ax_inset.set_xlim(0, 40)
    ax_inset.set_ylim(0, max(counts0.max(), counts1.max())+5)
    ax_inset.xaxis.set_major_locator(MultipleLocator(10))
    ax_inset.yaxis.set_major_locator(MultipleLocator(50))
    ax_inset.minorticks_on()
    ax_inset.set_xlabel(r'$G_\mathrm{c,max}$', fontsize=9, labelpad=1.5)
    ax_inset.tick_params(which='major', labelsize=8, length=2, width=.7, pad=1.5) 
    ax_inset.tick_params(which='minor', length=1.5, width=.5)

    # Scatter plot of CFI with respect to <n> for $\chi t= 0.8$.
    epsilon_start = 8
    n_ev = np.array([np.load(f'data/scan_epsilon__TwoPhotonDriving/n_ev__epsilon{i}.npy') for i in range(len(epsilon))]) # n_ev here is of shape (len(epsilon), n_samples, N).
    x = n_ev[epsilon_start:, :, N_index].reshape(-1) # x here is of shape (len(epsilon[:epsilon_start])*n_samples,).
    colors = np.repeat(epsilon[epsilon_start:], n_samples)

    y = result[epsilon_start:, :, N_index, 4].reshape(-1)
    sc = axs[0, 1].scatter(x, y, c=colors, cmap='viridis', s=.1, edgecolors='none')
    cbar = fig.colorbar(sc, ax=axs[0, 1], pad=-.12, location='top', shrink=.68, anchor=(1, 1))
    axs[0, 1].text(.15, 1.05, r'$\epsilon/\chi$', transform=axs[0, 1].transAxes)
    cbar.ax.xaxis.set_major_locator(MultipleLocator(20))
    cbar.ax.tick_params(which='major', length=1.5, pad=.5)
    axs[0, 1].set_yticks([0, 20000, 40000])
    axs[0, 1].set_yticklabels([0, 2, 4])
    axs[0, 1].text(-.2, .92, r'$\times 10^4$', transform=axs[0, 1].transAxes)
    axs[0, 1].set_xlabel(r'$\langle \hat{n} \rangle$')
    axs[0, 1].set_ylabel(r'$I_\mathrm{c,max}$')
    axs[0, 1].set_xlim(0, int(max(x)))
    axs[0, 1].set_ylim(0, max(y))
    axs[0, 1].minorticks_on()
    popt, _ = curve_fit(model, x, y, p0=[2, 2])
    a, b = popt
    print(f'optimal parameters for Ic: {a=:.2f}, {b=:.2f}.')
    axs[0, 1].plot(np.arange(1, int(max(x))+1), model(np.arange(1, int(max(x))+1), *popt), 'r--')
    axs[0, 1].text(.55, .15, rf'$I_\mathrm{{c,max}}\propto\langle \hat{{n}}\rangle^{{{b:.2f}}}$', rotation=45, color='r', transform=axs[0, 1].transAxes)
    axs[0, 1].plot(np.arange(1, int(max(x))+1), 4*np.arange(1, int(max(x))+1), color='k', linestyle='dashed') # SQL
    
    # thetab
    y = result[epsilon_start:, :, N_index, 2].reshape(-1)
    ax_inset = axs[0, 1].inset_axes([.035, .46, .51, .51])
    ax_inset.scatter(x, y, c=colors, cmap='viridis', s=.1, edgecolors='none')
    ax_inset.set_yscale('log')
    ax_inset.yaxis.tick_right()
    ax_inset.yaxis.set_label_position("right")
    ax_inset.set_xlim(min(x), max(x))
    ax_inset.set_ylim(min(y)-3e-4, max(y))
    ax_inset.xaxis.set_major_locator(MultipleLocator(50))
    ax_inset.minorticks_on()
    ax_inset.tick_params(which='major', labelsize=8, length=2, width=.7, pad=1.5) 
    ax_inset.tick_params(which='minor', length=1.5, width=.5)
    ax_inset.set_xlabel(r'$\langle\hat{n}\rangle$', fontsize=9, labelpad=.5)
    popt, _ = curve_fit(model, x, y, p0=[.5, -1],  sigma=1/x, absolute_sigma=True)
    a, b = popt
    print(f'optimal parameters for thetab: {a=:.2f}, {b=:.2f}.')
    ax_inset.plot(np.arange(int(min(x)), int(max(x))), model(np.arange(int(min(x)), int(max(x))), *popt), 'r--', linewidth=.7)

    N_stop = 30
    Gc = result[epsilon_start:, :, :N_stop, 1]
    data = [np.mean(Gc, axis=1), np.std(Gc, axis=1)]
    im = axs[1, 0].pcolormesh(np.arange(1, N_stop+1)*tau, epsilon[epsilon_start:], data[0], cmap='rainbow')
    cbar = fig.colorbar(im, ax=axs[1, 0], pad=-.12, location='top', shrink=.68, anchor=(1, 1))
    axs[1, 0].text(.07, 1.05, r'$\overline{G_\mathrm{c,max}}$', transform=axs[1, 0].transAxes)
    cbar.ax.xaxis.set_major_locator(MultipleLocator(30))
    cbar.ax.tick_params(which='major', direction='in', length=1.5, pad=.5)
    axs[1, 0].scatter(np.array([N_index+1, N_index+1])*tau, epsilon[epsilon_index], s=12, c=color)

    im = axs[1, 1].pcolormesh(np.arange(1, N_stop+1)*tau, epsilon[epsilon_start:], data[1], cmap='rainbow')
    cbar = fig.colorbar(im, ax=axs[1, 1], pad=-.12, location='top', shrink=.68, anchor=(1, 1))
    axs[1, 1].text(.06, 1.05, r'$\sigma_{G_\mathrm{c,max}}$', transform=axs[1, 1].transAxes)
    cbar.ax.xaxis.set_major_locator(MultipleLocator(5))
    cbar.ax.tick_params(which='major', direction='in', length=1.5, pad=.5)


    # $\epsilon/\chi = 20$ for different $\chi t$ in axs[0, 3].
    x = n_ev[epsilon_index[0], :, :N_stop].flatten() # x here is of shape: (n_samples*N_stop,)
    y1 = result[epsilon_index[0], :, :N_stop, 4].flatten()
    y2 = result[epsilon_index[0], :, :N_stop, 2].flatten()
    colors = np.tile(np.arange(1, N_stop+1)*tau, n_samples)

    # CFI
    scatter = axs[0, 2].scatter(x, y1, c=colors, cmap='viridis', s=.1, edgecolors='none')
    axs[0, 2].set_yticks([0, 5000, 10000, 15000])
    axs[0, 2].set_yticklabels([0, 5, 10, 15])
    axs[0, 2].text(-.2, .93, r'$\times 10^3$', transform=axs[0, 2].transAxes)
    axs[0, 2].set_xlabel(r'$\langle\hat{n}\rangle$')
    axs[0, 2].set_ylabel(r'$I_\mathrm{c,max}$')
    axs[0, 2].set_xlim(0, max(x)+1)
    axs[0, 2].set_ylim(0, max(y1)+1)
    axs[0, 2].minorticks_on()
    popt, _ = curve_fit(model, x, y1, p0=[2, 2])
    a, b = popt
    print(f'optimal parameters for Ic: {a=:.2f}, {b=:.2f}.')
    axs[0, 2].plot(np.arange(1, int(max(x))), model(np.arange(1, int(max(x))), *popt), 'r--')
    axs[0, 2].plot(np.arange(1, int(max(x))), 4*np.arange(1, int(max(x))), 'k--')
    cbar = fig.colorbar(scatter, ax=axs[0, 2], pad=-.12, location='top', shrink=.68, anchor=(1, 1))
    axs[0, 2].text(.2, 1.03, r'$\chi T$', transform=axs[0, 2].transAxes)
    cbar.ax.tick_params(which='major', direction='in', length=1.5, pad=.5)
    cbar.ax.xaxis.set_major_locator(MultipleLocator(1))
    
    # thetab
    ax_inset = axs[0, 2].inset_axes([.035, .52, .46, .46])
    ax_inset.scatter(x, y2, c=colors, cmap='viridis', s=.1, edgecolors='none')
    ax_inset.set_yscale('log')
    ax_inset.yaxis.tick_right()
    ax_inset.yaxis.set_label_position("right")
    ax_inset.set_xlim(min(x), max(x))
    ax_inset.set_ylim(min(y2)-3e-4, max(y2))
    ax_inset.xaxis.set_major_locator(MultipleLocator(25))
    ax_inset.minorticks_on()
    ax_inset.tick_params(which='major', labelsize=8, length=2, width=.7, pad=1.5)
    ax_inset.tick_params(which='minor', length=1.5, width=.5)
    ax_inset.set_xlabel(r'$\langle\hat{n}\rangle$', fontsize=9, labelpad=.5)
    popt, _ = curve_fit(model, x, y2, p0=[.5, -1],  sigma=1/x, absolute_sigma=True)
    a, b = popt
    print(f'optimal parameters for thetab: {a=:.2f}, {b=:.2f}.')
    ax_inset.plot(np.arange(1, int(max(x))), model(np.arange(1, int(max(x))), *popt), 'r--', linewidth=.7)

    # G_SR = Gc - 2*sigma_Gc
    Gc = result[epsilon_start:, :, :N_stop, 1]
    G_SR = Gc.mean(axis=1) - 2*Gc.std(axis=1)
    cm = axs[1, 2].pcolormesh(np.arange(1, N_stop+1)*tau, epsilon[epsilon_start:], G_SR, cmap='rainbow')
    cbar = fig.colorbar(cm, ax=axs[1, 2], pad=-.12, location='top', shrink=.68, anchor=(1, 1))
    cbar.ax.tick_params(which='major', direction='in', length=1.5, pad=.5)
    cbar.ax.xaxis.set_major_locator(MultipleLocator(20))
    axs[1, 2].xaxis.set_major_locator(MultipleLocator(1))
    axs[1, 2].minorticks_on()
    axs[1, 2].set_yscale('log')
    axs[1, 2].set_ylabel(r'$\epsilon/\chi$')
    axs[1, 2].set_xlabel(r'$\chi t$')
    axs[1, 2].text(0.15, 1.03, r'$G_\mathrm{SR}$', transform=axs[1, 2].transAxes)

    CS = axs[1, 0].contour(np.arange(1, N_stop+1)*tau, epsilon[epsilon_start:], data[0], levels=[10, 20, 40], colors='k', linewidths=.5)
    axs[1, 0].clabel(CS, fontsize=9, manual=[(1.6, 8), (1.8, 15), (2, 30)])
    CS = axs[1, 1].contour(np.arange(1, N_stop+1)*tau, epsilon[epsilon_start:], data[1], levels=[3, 5, 7], colors='k', linewidths=.5)
    axs[1, 1].clabel(CS, fontsize=9, manual=[(2, 10), (2, 20), (2, 30)])
    CS = axs[1, 2].contour(np.arange(1, N_stop+1)*tau, epsilon[epsilon_start:], G_SR, levels=[3, 5, 10, 20, 40], colors='k', linewidths=.5)
    axs[1, 2].clabel(CS, fontsize=9, manual=[(.8, 8), (1, 10), (1.2, 15), (1.6, 22), (2, 40)])

    text = [[r'\textbf{a}', r'\textbf{b}', r'\textbf{c}'], [r'\textbf{d}', r'\textbf{e}', r'\textbf{f}']]
    for i in range(2):
        for j in range(3):
            axs[i, j].text(-.26, 1.05, text[i][j], transform=axs[i, j].transAxes)
            if i==1:
                axs[1, j].xaxis.set_major_locator(MultipleLocator(1))
                axs[1, j].minorticks_on()
                axs[1, j].set_yscale('log')
                axs[1, j].set_xlabel(r'$\chi T$')
                axs[1, j].set_ylabel(r'$\epsilon/\chi$')

    fig.savefig('figures/figS6.pdf')


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--scan_epsilon', action='store_true')
    parser.add_argument('--plot_figS6', action='store_true')
    args = parser.parse_args()

    if args.scan_epsilon:
        scan_epsilon()

    if args.plot_figS6:
        plot_figS6()