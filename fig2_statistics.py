from functools import partial
from pathlib import Path
from argparse import ArgumentParser
import warnings

import numpy as np
from qutip import destroy, basis, expect, num, wigner, displace
from scipy.optimize import curve_fit
from scipy.stats import norm
from mpi4py.futures import MPIPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FuncFormatter
import matplotlib.colors as mcolors

from utils import timing, mpl_rcParams, figwidth_dc, trim_zeros
mpl.rcParams.update(mpl_rcParams)

Path('data/scan_epsilon__SinglePhotonDriving').mkdir(exist_ok=True)
Path('figures').mkdir(exist_ok=True)
#------------------------------------------------------------------------------------------------------------------------------------#


def forward(dim, tau, driving_type, driving_list, alpha=0):
    u1_list, u2_list = driving_list # driving_list here is of shape (2, N).
    a = destroy(dim)
    num_op = num(dim)
    H0 = num_op**2 # $\chi=1$
    match driving_type:
        case 'SinglePhotonDriving':
            H1 = a.dag() + a
            H2 = a.dag() - a
        case 'TwoPhotonDriving':
            H1 = a.dag()**2 + a**2
            H2 = a.dag()**2 - a**2
        case _:
            raise ValueError('driving_type could only be one of "SinglePhotonDriving" or "TwoPhotonDriving".')

    N = driving_list.shape[1]
    psi0 = np.empty(N, dtype=object)
    n_ev = np.zeros(N)
    temp = displace(dim, alpha) * basis(dim, 0)
    for i in range(N):
        H = H0 + u1_list[i]*H1 + 1j*u2_list[i]*H2
        temp = (-1j*H*tau).expm() * temp
        psi0[i] = temp
        n_ev[i] = expect(num_op, temp)

        n_std = np.sqrt(expect(num_op**2, temp) - n_ev[i]**2)
        threshold = n_ev[i] + 3*n_std
        if threshold > dim:
            warnings.warn(f"Truncation may be inaccurate. Try a truncation number larger than {int(threshold)}.")
    return [n_ev, psi0]

def Gq_Gc_thetab_QFI_CFI(psi0, epsilon_dp=1e-3): # given psi0, return [Gq, Gc, theta_b, QFI, CFI]
    dim = psi0.shape[0]
    psi0 = psi0.data_as('ndarray').flatten()
    n = np.arange(dim)
    n_ev = (psi0.conjugate()@(n*psi0)).real
    var_n = (psi0.conjugate()@(n**2 * psi0)).real - n_ev**2
    QFI = 4*var_n
    Gq = var_n / n_ev

    CFI = 0
    Gc = 0
    delta = 5e-4
    i_max = 6000
    for i in range(1, i_max+1):
        thetab = i*delta
        expn = np.exp(-1j*thetab*n)
        psi_theta = expn * psi0
        temp = psi0.conjugate() @ psi_theta
        p0 = (temp*temp.conjugate()).real
        dp0 = 2*(psi0.conjugate()@(-1j*n*psi_theta) * temp.conjugate()).real                            # derivative of p0 with respect to theta.
        if epsilon_dp!=0:                                                    # depolarizing noise with strength epsilon_dp.
            p0 = (1-epsilon_dp)*p0 + epsilon_dp/dim
            dp0 = (1-epsilon_dp)*dp0
        
        cfi = 0
        temp = dp0**2
        if p0 != 0:
            cfi += temp / p0
        if p0 != 1:
            cfi += temp / (1-p0)
        gc = cfi / (4*n_ev)
        if gc<Gc:
            break        
        if gc>Gc:
            Gc = gc
            CFI = cfi
    if i==i_max:
        warnings.warn('$Try increase i_max$.')
    return [Gq, Gc, thetab, QFI, CFI]

def compute_Gc_vs_theta0(psi0, theta0=np.linspace(0, .15, 151), epsilon_dp=1e-3):
    dim = psi0.shape[0]
    psi0 = psi0.data_as('ndarray').flatten()
    n = np.arange(dim)
    n_ev = (psi0.conjugate()@(n*psi0)).real

    Gc = np.empty_like(theta0)
    for i in range(len(theta0)):
        expn = np.exp(-1j*theta0[i]*n)
        psi_theta = expn * psi0
        temp1 = psi0.conjugate() @ psi_theta
        p0 = (temp1*temp1.conjugate()).real
        temp2 = psi0.conjugate()@(-1j*n*psi_theta) * temp1.conjugate()
        dp0 = (temp2 + temp2.conjugate()).real                            # derivative of p0 with respect to theta.
        if epsilon_dp!=0:                                                    # depolarizing noise with strength epsilon_dp.
            p0 = (1-epsilon_dp)*p0 + epsilon_dp/dim
            dp0 = (1-epsilon_dp)*dp0
        
        CFI = 0
        temp = dp0**2
        if p0 != 0:
            CFI += temp / p0
        if p0 != 1:
            CFI += temp / (1-p0)
        Gc[i] = CFI / (4*n_ev)
    return Gc

@timing
def scan_epsilon():
    dim = 1050
    tau = .1
    N = 70 # number of time steps
    epsilon = np.logspace(1, 3, 21)
    n_samples = 1000

    rng = np.random.default_rng(2025)
    driving_list = np.zeros((len(epsilon), n_samples, 2, N))
    for i in range(len(epsilon)):
        driving_list[i] = rng.uniform(-epsilon[i], epsilon[i], size=(n_samples, 2, N))
    np.save('data/scan_epsilon__SinglePhotonDriving/driving_list.npy', driving_list)

    with MPIPoolExecutor() as executor:
        for i in tqdm(range(len(epsilon)), desc='forward'):
            n_ev = np.zeros((n_samples, N))
            psi0 = np.empty((n_samples, N), dtype=object)            
            result = executor.map(partial(forward, dim, tau, 'SinglePhotonDriving'), driving_list[i])
            for j, res in enumerate(result):
                n_ev[j], psi0[j] = res

            np.save(f'data/scan_epsilon__SinglePhotonDriving/n_ev__epsilon{i}.npy', n_ev)
            np.save(f'data/scan_epsilon__SinglePhotonDriving/psi0__epsilon{i}.npy', psi0)


        result = np.empty((len(epsilon), n_samples, N, 5))
        for i in tqdm(range(len(epsilon)), desc='Gq_Gc_thetab_QFI_CFI'):
            psi0 = np.load(f'data/scan_epsilon__SinglePhotonDriving/psi0__epsilon{i}.npy', allow_pickle=True) # psi0 here is of shape (n_samples, N) with dtype=object.
            temp = list(executor.map(Gq_Gc_thetab_QFI_CFI, psi0.reshape(-1)))
            result[i] = np.array(temp).reshape((n_samples, N, 5))
        np.save('data/scan_epsilon__SinglePhotonDriving/Gq_Gc_thetab_QFI_CFI.npy', result)

def model(n_ev, a, b):
    return a * n_ev**b

def p0_Gc(psi0, theta0, epsilon_dp=1e-3):
    dim = psi0.shape[0]
    psi0 = psi0.data_as('ndarray').flatten()
    n = np.arange(dim)
    expn = np.exp(-1j*theta0*n)

    psi_theta = expn * psi0
    temp1 = psi0.conjugate() @ psi_theta
    p0 = (temp1*temp1.conjugate()).real

    temp2 = psi0.conjugate()@(-1j*n*psi_theta) * temp1.conjugate()
    dp0 = (temp2 + temp2.conjugate()).real                            # derivative of p0 with respect to theta.
    
    if epsilon_dp!=0:                                                    # depolarizing noise with strength epsilon.
        p0 = (1-epsilon_dp)*p0 + epsilon_dp/dim
        dp0 = (1-epsilon_dp)*dp0

    CFI = 0
    temp = dp0**2
    if p0 != 0:
        CFI += temp / p0
    if p0 != 1:
        CFI += temp / (1-p0)
    
    n_ev = (psi0.conjugate()@(n*psi0)).real
    Gc = CFI / (4*n_ev)
    return [p0, Gc]

@timing
def plot_fig2():
    tau = .1
    epsilon = np.logspace(1, 3, 21)
    n_samples = 1000
    epsilon_index = [10, 13] # corresponds to $\epsilon/\chi = [100, 200]$.
    N_index = 19 # corresponds to $\chi t=2$.
    result = np.load('data/scan_epsilon__SinglePhotonDriving/Gq_Gc_thetab_QFI_CFI.npy') # result is of shape (len(epsilon), n_samples, N, 5).

    fig, axs = plt.subplots(2, 3, layout='constrained', figsize=(figwidth_dc, figwidth_dc*.58))

    # plot one example
    sample_index = 1
    theta0 = np.linspace(0, .2, 201)
    psi0 = np.load(f'data/scan_epsilon__SinglePhotonDriving/psi0__epsilon{epsilon_index[0]}.npy', allow_pickle=True)[sample_index, N_index]
    n_ev = np.load(f'data/scan_epsilon__SinglePhotonDriving/n_ev__epsilon{epsilon_index[0]}.npy')[sample_index, N_index]
    Gq = result[epsilon_index[0], sample_index, N_index, 0]
    print(f'Average number of photons of the sample state is: {n_ev}. The QFI of the sample state is: {Gq}.')
    
    data = np.array([p0_Gc(psi0, theta0) for theta0 in theta0])
    np.save('data/fig2_p0_Gc.npy', data)

    xvec = np.linspace(-1, 1, 201)
    yvec = np.linspace(-1, 1, 201)
    W = wigner(psi0, xvec, yvec)
    np.save('data/fig2_W.npy', W)

    data = np.load('data/fig2_p0_Gc.npy')
    W = np.load('data/fig2_W.npy')
    vmax = np.abs(W).max()
    ylabel = [r'$p_0(\theta_0)$', r'$G_\mathrm{c}(\theta_0)$']
    axs[0, 0].plot(theta0, data[:, 0])
    axs[1, 0].plot(theta0, data[:, 1])
    axs[1, 0].hlines(Gq, xmin=0, xmax=theta0.max(), colors='r', linestyles='dashed')
    axs[1, 0].hlines(1, xmin=0, xmax=theta0.max(), colors='k', linestyles='dashed')
    for j in range(2):
        axs[j, 0].xaxis.set_major_locator(MultipleLocator(0.05))
        axs[j, 0].minorticks_on()
        axs[j, 0].set_xlabel(r'$\theta_0$')
        axs[j, 0].set_ylabel(ylabel[j], labelpad=2)
        axs[j, 0].set_xlim(-1e-3, .17)
        axs[j, 0].xaxis.set_major_formatter(FuncFormatter(trim_zeros))
        axs[j, 0].yaxis.set_major_formatter(FuncFormatter(trim_zeros))
    axs[0, 0].set_ylim(0, 1)
    axs[1, 0].set_ylim(0, 23)

    ax_inset = axs[0, 0].inset_axes([.46, .4, .56, .56])
    ax_inset.pcolormesh(xvec, yvec, W, vmin=-vmax, vmax=vmax, cmap='bwr')
    ax_inset.xaxis.set_major_locator(MultipleLocator(1))
    ax_inset.yaxis.set_major_locator(MultipleLocator(1))
    ax_inset.set_aspect('equal')   
    ax_inset.tick_params(which='major', labelsize=8, length=2, width=.7, pad=1.5) 
    ax_inset.tick_params(which='minor', length=1.5, width=.5)


    # plot statistics
    data = [np.load(f'data/scan_epsilon__SinglePhotonDriving/psi0__epsilon{i}.npy', allow_pickle=True)[:, N_index] for i in epsilon_index] # data here is of shape (2, n_samples).
    theta0 = np.linspace(0, .1, 101)
    Gc_theta0 = np.array([[compute_Gc_vs_theta0(psi0_, theta0) for psi0_ in psi0] for psi0 in data]) # Gc_theta0 here is of shape (2, n_samples, len(theta0))
    Gc_theta0_ave = [np.mean(A, axis=0) for A in Gc_theta0]
    Gc_theta0_std = [np.std(A, axis=0) for A in Gc_theta0]
    color = ['lightgreen', 'pink']
    for i in range(len(epsilon_index)):
        axs[0, 1].plot(theta0, Gc_theta0_ave[i], color=color[i])
        axs[0, 1].fill_between(theta0, Gc_theta0_ave[i]-Gc_theta0_std[i], Gc_theta0_ave[i]+Gc_theta0_std[i], color=color[i], edgecolor='none', alpha=.7)
    axs[0, 1].set_xlim(-1e-3, theta0.max()+1e-3)
    axs[0, 1].set_ylim(0, np.max(Gc_theta0_ave[1]+Gc_theta0_std[1]+1))
    axs[0, 1].minorticks_on()
    axs[0, 1].xaxis.set_major_locator(MultipleLocator(0.05))
    axs[0, 1].yaxis.set_major_locator(MultipleLocator(10))
    axs[0, 1].set_xlabel(r'$\theta_0$')
    axs[0, 1].set_ylabel(r'$G_\mathrm{c}(\theta_0)$')
    axs[0, 1].xaxis.set_major_formatter(FuncFormatter(trim_zeros))

    sample0 = result[epsilon_index[0], :, N_index, 1]
    sample1 = result[epsilon_index[1], :, N_index, 1]
    ax_inset = axs[0, 1].inset_axes([.42, .42, .54, .54])
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

    # Scatter plot of CFI with respect to <n> for $\epsilon/\chi = 100$.
    epsilon_start = 6
    n_ev = np.array([np.load(f'data/scan_epsilon__SinglePhotonDriving/n_ev__epsilon{i}.npy') for i in range(len(epsilon))]) # n_ev here is of shape (len(epsilon), n_samples, N).
    x = n_ev[epsilon_start:, :, N_index].reshape(-1) # x here is of shape (len(epsilon[:epsilon_start])*n_samples,).
    colors = np.repeat(epsilon[epsilon_start:], n_samples)

    y = result[epsilon_start:, :, N_index, 4].reshape(-1)
    sc = axs[1, 1].scatter(x, y, c=colors, cmap='viridis', norm='log', s=.1, edgecolors='none')
    cbar = fig.colorbar(sc, ax=axs[1, 1], pad=-.11, location='top', shrink=.68, anchor=(1, 1))
    axs[1, 1].text(.15, 1.05, r'$\epsilon/\chi$', transform=axs[1, 1].transAxes)
    cbar.ax.tick_params(which='major', direction='in', length=2.5, pad=0)
    cbar.ax.tick_params(which='minor', direction='in', length=1.5)
    axs[1, 1].set_yticks([0, 30000, 60000, 90000])
    axs[1, 1].set_yticklabels([0, 3, 6, 9])
    axs[1, 1].text(-.18, .92, r'$\times 10^4$', transform=axs[1, 1].transAxes)
    axs[1, 1].set_xlabel(r'$\langle \hat{n} \rangle$')
    axs[1, 1].set_ylabel(r'$I_\mathrm{c,max}$')
    axs[1, 1].set_xlim(0, int(max(x)))
    axs[1, 1].set_ylim(0, max(y))
    axs[1, 1].minorticks_on()
    popt, _ = curve_fit(model, x, y, p0=[1, 1.5])
    a, b = popt
    print(f'optimal parameters for Ic: {a=:.2f}, {b=:.2f}.')
    axs[1, 1].plot(np.arange(1, int(max(x))+1), model(np.arange(1, int(max(x))+1), *popt), 'r--')
    axs[1, 1].text(.55, .15, rf'$I_\mathrm{{c,max}}\propto\langle \hat{{n}}\rangle^{{{b:.2f}}}$', rotation=50, color='r', transform=axs[1, 1].transAxes)
    axs[1, 1].plot(np.arange(1, int(max(x))+1), 4*np.arange(1, int(max(x))+1), color='k', linestyle='dashed') # SQL
    
    # thetab
    y = result[epsilon_start:, :, N_index, 2].reshape(-1)
    ax_inset = axs[1, 1].inset_axes([.04, .46, .51, .51])
    ax_inset.scatter(x, y, c=colors, cmap='viridis', s=.1, edgecolors='none')
    ax_inset.set_yscale('log')
    ax_inset.yaxis.tick_right()
    ax_inset.yaxis.set_label_position("right")
    ax_inset.set_xlim(min(x), max(x))
    ax_inset.set_ylim(min(y)-3e-4, max(y))
    ax_inset.xaxis.set_major_locator(MultipleLocator(100))
    ax_inset.minorticks_on()
    ax_inset.tick_params(which='major', labelsize=8, length=2, width=.7, pad=1.5) 
    ax_inset.tick_params(which='minor', length=1.5, width=.5)
    ax_inset.set_xlabel(r'$\langle\hat{n}\rangle$', fontsize=9, labelpad=.5)
    popt, _ = curve_fit(model, x, y, p0=[.5, -.5],  sigma=1/x, absolute_sigma=True)
    a, b = popt
    print(f'optimal parameters for thetab: {a=:.2f}, {b=:.2f}.')
    ax_inset.plot(np.arange(int(min(x)), int(max(x))), model(np.arange(int(min(x)), int(max(x))), *popt), 'r--', linewidth=.7)

    N_stop = 30
    Gc = result[epsilon_start:, :, :N_stop, 1]
    data = [np.mean(Gc, axis=1), np.std(Gc, axis=1)]
    im = axs[0, 2].pcolormesh(np.arange(1, N_stop+1)*tau, epsilon[epsilon_start:], data[0], cmap='rainbow')
    cbar = fig.colorbar(im, ax=axs[0, 2], pad=-.11, location='top', shrink=.68, anchor=(1, 1))
    axs[0, 2].text(.08, 1.03, r'$\overline{G_\mathrm{c,max}}$', transform=axs[0, 2].transAxes)
    cbar.ax.xaxis.set_major_locator(MultipleLocator(30))
    cbar.ax.tick_params(direction='in', which='major', length=2, pad=0)
    im = axs[1, 2].pcolormesh(np.arange(1, N_stop+1)*tau, epsilon[epsilon_start:], data[1], cmap='rainbow')
    cbar = fig.colorbar(im, ax=axs[1, 2], pad=-.11, location='top', shrink=.68, anchor=(1, 1))
    axs[1, 2].text(.08, 1.05, r'$\sigma_{G_\mathrm{c,max}}$', transform=axs[1, 2].transAxes)
    cbar.ax.xaxis.set_major_locator(MultipleLocator(5))
    cbar.ax.tick_params(direction='in', which='major', length=2, pad=0)
    for i in range(2):
        axs[i, 2].xaxis.set_major_locator(MultipleLocator(1))
        axs[i, 2].minorticks_on()
        axs[i, 2].set_yscale('log')
        axs[i, 2].set_xlabel(r'$\chi T$')
        axs[i, 2].set_ylabel(r'$\epsilon/\chi$')

    CS = axs[0, 2].contour(np.arange(1, N_stop+1)*tau, epsilon[epsilon_start:], data[0], levels=[10, 20, 40, 70], colors='k', linewidths=.5)
    axs[0, 2].clabel(CS, manual=[(1.6, 50), (2, 150), (2.2, 300), (2.4, 500)], fontsize=9)
    axs[0, 2].scatter(np.array([N_index+1, N_index+1])*tau, epsilon[epsilon_index], s=12, c=color)

    CS = axs[1, 2].contour(np.arange(1, N_stop+1)*tau, epsilon[epsilon_start:], data[1], levels=[3, 5, 7], colors='k', linewidths=.5)
    axs[1, 2].clabel(CS, manual=[(2.4, 100), (2.4, 300), (2.4, 600)], fontsize=9)

    axs[0, 0].text(-.2, 1.05, r'\textbf{a}', transform=axs[0, 0].transAxes)
    axs[1, 0].text(-.2, 1.05, r'\textbf{b}', transform=axs[1, 0].transAxes)
    axs[0, 1].text(-.19, 1.05, r'\textbf{c}', transform=axs[0, 1].transAxes)
    axs[1, 1].text(-.19, 1.05, r'\textbf{d}', transform=axs[1, 1].transAxes)
    axs[0, 2].text(-.2, 1.05, r'\textbf{e}', transform=axs[0, 2].transAxes)
    axs[1, 2].text(-.2, 1.05, r'\textbf{f}', transform=axs[1, 2].transAxes)
    fig.savefig('figures/fig2.pdf')


@timing
def plot_figS2():
    epsilon = np.logspace(1, 3, 21)
    tau = .1
    epsilon_start = 6
    N_stop = 30
    result = np.load('data/scan_epsilon__SinglePhotonDriving/Gq_Gc_thetab_QFI_CFI.npy') # result is of shape (len(epsilon), n_samples, N, 5).
    Gc = result[epsilon_start:, :, :N_stop, 1]    
    thetab = result[epsilon_start:, :, :N_stop, 2]
    n_ev = np.array([np.load(f'data/scan_epsilon__SinglePhotonDriving/n_ev__epsilon{i}.npy') for i in range(len(epsilon))])[epsilon_start:, :, :N_stop] # n_ev here is of shape (len(epsilon[epsilon_start:]), n_samples, N_stop)
    data = [Gc.mean(axis=1)-2*Gc.std(axis=1)] + [np.mean(A, axis=1) for A in [n_ev, thetab]]
    norm = [None, None, 'log']
    text = [r'\textbf{a}', r'\textbf{b}', r'\textbf{c}']
    label = [r'$G_\mathrm{SR}$', r'$\langle \hat{n}\rangle$', r'$\theta_\mathrm{b}$']
    text_x = [.17, .2, .21]
    
    fig, axs = plt.subplots(1, 3, layout='constrained', figsize=(figwidth_dc, figwidth_dc*.28))
    for i in range(3):
        cm = axs[i].pcolormesh(np.arange(1, N_stop+1)*tau, epsilon[epsilon_start:], data[i], cmap='rainbow', norm=norm[i])
        cbar = fig.colorbar(cm, ax=axs[i], pad=-.12, location='top', shrink=.68, anchor=(1, 1))
        cbar.ax.tick_params(which='minor', direction='in', length=1)
        cbar.ax.tick_params(which='major', direction='in', length=1.5, pad=0)
        cbar.ax.minorticks_on()
        axs[i].xaxis.set_major_locator(MultipleLocator(1))
        axs[i].minorticks_on()
        axs[i].set_yscale('log')
        axs[i].set_ylabel(r'$\epsilon/\chi$')
        axs[i].set_xlabel(r'$\chi T$')
        axs[i].text(-.24, 1.05, text[i], transform=axs[i].transAxes)
        axs[i].text(text_x[i], 1.03, label[i], transform=axs[i].transAxes)
        
    CS = axs[0].contour(np.arange(1, N_stop+1)*tau, epsilon[epsilon_start:], data[0], levels=[1, 5, 10, 20, 40], colors='k', linewidths=.5)
    axs[0].clabel(CS, manual=[(.8, 70), (1.2, 90), (1.8, 150), (2, 200), (2.2, 400)], fontsize=9)
    CS = axs[1].contour(np.arange(1, N_stop+1)*tau, epsilon[epsilon_start:], data[1], levels=[20, 30, 50, 100], colors='k', linewidths=.5)
    axs[1].clabel(CS, manual=[(1, 60), (1.2, 90), (1.4, 170), (1.6, 400)], fontsize=9)
    CS = axs[2].contour(np.arange(1, N_stop+1)*tau, epsilon[epsilon_start:], data[2], levels=[.003, .005, .01], colors='k', linewidths=.5)
    axs[2].clabel(CS, manual=[(2, 500), (1.8, 300), (1.4, 100)], fmt=lambda x: f"{x:g}", fontsize=9)
    fig.savefig('figures/figS2.pdf')

@timing
def plot_figS3():
    epsilon = np.logspace(1, 3, 21)
    tau = .1
    n_samples = 1000
    N_stop = 30
    epsilon_start = 6

    result = np.load('data/scan_epsilon__SinglePhotonDriving/Gq_Gc_thetab_QFI_CFI.npy') # result is of shape (len(epsilon), n_samples, N, 5).
    Gc = result[:, :, :, 1]
    CFI = result[:, :, :, 4]
    thetab = result[:, :, :, 2]
    n_ev = np.array([np.load(f'data/scan_epsilon__SinglePhotonDriving/n_ev__epsilon{i}.npy') for i in range(len(epsilon))])

    fig, axs = plt.subplots(1, 2, layout='constrained', figsize=(figwidth_dc*.67, figwidth_dc*.28))

    # $\epsilon/\chi = 100$ for different $\chi t$ in axs[0] and axs[1].
    epsilon_index = 10  # corresponds to $\epsilon/\chi = 100$.
    x = n_ev[epsilon_index, :, :N_stop].flatten() # x here is of shape: (n_samples*N_stop,)
    y1 = CFI[epsilon_index, :, :N_stop].flatten()
    y2 = thetab[epsilon_index, :, :N_stop].flatten()
    colors = np.tile(np.arange(1, N_stop+1)*tau, n_samples)

    # CFI
    scatter = axs[0].scatter(x, y1, c=colors, cmap='viridis', s=.1, edgecolors='none')
    axs[0].set_yticks([0, 2000, 4000])
    axs[0].set_yticklabels([0, 2, 4])
    axs[0].text(-.18, .91, r'$\times 10^3$', transform=axs[0].transAxes)
    axs[0].set_xlim(0, max(x)+1)
    axs[0].set_ylim(0, max(y1)+1)
    popt, _ = curve_fit(model, x, y1, p0=[1, -1])
    a, b = popt
    print(f'optimal parameters for Ic: {a=:.2f}, {b=:.2f}.')
    axs[0].plot(np.arange(1, int(max(x))), model(np.arange(1, int(max(x))), *popt), 'r--')
    axs[0].plot(np.arange(1, int(max(x))), 4*np.arange(1, int(max(x))), 'k--')
    cbar = fig.colorbar(scatter, ax=axs[0], pad=-.12, location='top', shrink=.68, anchor=(1, 1))
    axs[0].text(.2, 1.03, r'$\chi T$', transform=axs[0].transAxes)
    cbar.ax.tick_params(which='major', direction='in', length=1.5, pad=0)
    cbar.ax.tick_params(which='minor', direction='in', length=1)
    cbar.ax.minorticks_on()
    cbar.ax.xaxis.set_major_locator(MultipleLocator(1))


    # thetab
    ax_inset = axs[0].inset_axes([.035, .525, .4, .46])
    ax_inset.scatter(x, y2, c=colors, cmap='viridis', s=.1, edgecolors='none')
    ax_inset.set_yscale('log')
    ax_inset.yaxis.tick_right()
    ax_inset.yaxis.set_label_position("right")
    ax_inset.set_xlim(0, max(x))
    ax_inset.xaxis.set_major_locator(MultipleLocator(20))
    ax_inset.minorticks_on()
    ax_inset.tick_params(which='major', direction='in', labelsize=8, length=1.5, width=.6, pad=1)
    ax_inset.tick_params(which='minor', direction='in', length=1, width=.4)
    ax_inset.set_xlabel(r'$\langle\hat{n}\rangle$', fontsize=9, labelpad=.5)
    popt, _ = curve_fit(model, x, y2, p0=[.5, -.5],  sigma=1/x, absolute_sigma=True)
    a, b = popt
    print(f'optimal parameters for thetab: {a=:.2f}, {b=:.2f}.')
    ax_inset.plot(np.arange(1, int(max(x))), model(np.arange(1, int(max(x))), *popt), 'r--', linewidth=.7)

    # $sigma_{Gc}$
    Gc_std = Gc[epsilon_start:, :, :N_stop].std(axis=1) # Gc_std here is of shape (len(epsilon[epsilon_start:]), N_stop).
    cmap = plt.get_cmap('viridis')
    norm = mcolors.LogNorm(vmin=min(epsilon[epsilon_start:]), vmax=max(epsilon[epsilon_start:]))
    colors = cmap(norm(epsilon[epsilon_start:]))  # colors now has shape (len(epsilon[epsilon_start:]), 4)
    for i in range(0, len(epsilon[epsilon_start:])):
        axs[1].plot(np.arange(1, N_stop+1)*tau, Gc_std[i], color=colors[i], linewidth=.8)
    axs[1].set_xlim(0, N_stop*tau)
    axs[1].set_ylim(0, Gc_std.max()+1)        
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axs[1], pad=-.12, location='top', shrink=.68, anchor=(1, 1))
    cbar.ax.tick_params(which='major', direction='in', length=1.5, pad=0)
    cbar.ax.tick_params(which='minor', direction='in', length=1)
    axs[1].text(.17, 1.03, r'$\epsilon/\chi$', transform=axs[1].transAxes)
   
    
    text = [r'\textbf{a}', r'\textbf{b}']
    xlabel = [r'$\langle \hat{n} \rangle$', r'$\chi T$']
    ylabel = [r'$I_\mathrm{c,max}$', r'$\sigma_{G_\mathrm{c,max}}$']
    for i in range(2):
        axs[i].set_xlabel(xlabel[i])
        axs[i].set_ylabel(ylabel[i])
        axs[i].minorticks_on()
        axs[i].text(-.2, 1.05, text[i], transform=axs[i].transAxes)

    fig.savefig('figures/figS3.pdf')


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--scan_epsilon', action='store_true')
    parser.add_argument('--plot_fig2', action='store_true')
    parser.add_argument('--plot_figS2', action='store_true')
    parser.add_argument('--plot_figS3', action='store_true')
    args = parser.parse_args()

    if args.scan_epsilon:
        scan_epsilon()
    
    if args.plot_fig2:
        plot_fig2()

    if args.plot_figS2:
        plot_figS2()

    if args.plot_figS3:
        plot_figS3()