from functools import partial
from argparse import ArgumentParser
from pathlib import Path
import warnings

import numpy as np
from qutip import destroy, num, basis, qeye, expect, fock_dm
from mpi4py.futures import MPIPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter, MultipleLocator
mpl.rcParams["mathtext.fontset"] = 'cm'

from fig2_statistics import compute_Gc_vs_theta0
from utils import timing, mpl_rcParams, figwidth_dc, trim_zeros
mpl.rcParams.update(mpl_rcParams)

Path('data').mkdir(exist_ok=True)
Path('figures').mkdir(exist_ok=True)
#------------------------------------------------------------------------------------------------------------------------------------#

def forward(dim, tau, driving_list):
    u1_list, u2_list = driving_list # driving_list here is of shape (2, N).
    a = destroy(dim)
    num_op = num(dim)
    H0 = num_op**2
    H1 = a.dag() + a
    H2 = a.dag() - a

    N = driving_list.shape[1]
    U = qeye(dim)
    for i in range(N):
        H = H0 + u1_list[i]*H1 + 1j*u2_list[i]*H2
        U = (-1j*H*tau).expm() * U

    psi0 = U * basis(dim, 0)
    nbar = expect(num_op, psi0)
    n_std = np.sqrt(expect(num_op**2, psi0) - nbar**2)
    threshold = nbar + 3*n_std
    if threshold > dim:
        warnings.warn(f"Truncation may be inaccurate. Try a truncation number larger than {int(threshold)}.")
    return psi0

@timing
def depolarizing_noise():
    dim = 170
    tau = .1
    epsilon_index = 10 # corresponds to $\epsilon/\chi=100$.
    N = 15 # corresponds to $\chi t=1.5$.
    driving_list = np.load('data/scan_epsilon__SinglePhotonDriving/driving_list.npy')[epsilon_index, :, :, :N] # driving_list here is of shape (n_samples, 2, N).
    n_samples = driving_list.shape[0]
    theta0 = np.linspace(0, .1, 201)

    with MPIPoolExecutor() as executor:
        psi0 = list(tqdm(executor.map(partial(forward, dim, tau), driving_list), total=n_samples))

        epsilon_dp = [0, 1e-3, 1e-2, 1e-1]
        Gc = []
        for i in tqdm(range(len(epsilon_dp))):
            Gc.append(list(executor.map(partial(compute_Gc_vs_theta0, theta0=theta0, epsilon_dp=epsilon_dp[i]), psi0)))
        np.save('data/Gc_depolarizing.npy', Gc) # Gc is of shape (len(epsilon_dp), n_samples, len(theta0)).


def gc_of_theta0(driving_list, n_fock, theta0, tau, dim, epsilon_dp=1e-3, only_gq=False):
    u1_list, u2_list = driving_list
    N = driving_list.shape[1]

    a = destroy(dim)
    H0 = (a.dag()*a)**2
    H1 = a.dag() + a
    H2 = a.dag() - a
    num_op = num(dim)
    vac = basis(dim, 0)

    # evolution
    U = qeye(dim)
    for i in range(N):
        H = H0 + u1_list[i]*H1 + 1j*u2_list[i]*H2
        U = (-1j*H*tau).expm() * U
    
    psi0 = U * vac
    nbar = expect(num_op, psi0)
    n2_bar = expect(num_op**2, psi0)
    n_std = np.sqrt(n2_bar - nbar**2)
    threshold = nbar+3*n_std
    if threshold > dim:
        warnings.warn(f"Truncation may be inaccurate. Try a truncation number larger than {int(threshold)}.")
    
    if only_gq:
        Gq = n_std**2 / nbar
        return Gq

    psi_theta = (-1j*theta0*num_op).expm() * psi0
    psi_f = U.dag() * psi_theta

    # measurement
    Gc = np.zeros(n_fock)
    for n in range(n_fock):
        p_list = np.zeros(n+2)
        dp_list = np.zeros(n+2)
        for i in range(n+1):
            p_list[i] = expect(fock_dm(dim, i), psi_f)
            dp_list[i] = 2*(basis(dim, i).dag()*U.dag()*(-1j*num_op)*psi_theta * psi_f.dag()*basis(dim, i)).real

        if epsilon_dp!=0:
            p_list[:n+1] = (1-epsilon_dp)*p_list[:n+1] + epsilon_dp/dim
            dp_list[:n+1] = (1-epsilon_dp)*dp_list[:n+1]
        
        p_list[n+1] = 1 - p_list[:n+1].sum()
        dp_list[n+1] = - dp_list[:n+1].sum()

        CFI = 0
        for i in range(n+2):
            if p_list[i]>0:
                CFI += dp_list[i]**2 / p_list[i]
        Gc[n] = CFI / (4*nbar)
    return Gc
    
def gc(n_fock, theta0, tau, dim, driving_list):
    return np.max([gc_of_theta0(driving_list, n_fock, theta0_, tau, dim) for theta0_ in theta0], axis=0)

@timing
def measurement():
    tau = .1
    dim = 170
    epsilon_index = 10
    N = 15
    n_samples = 100
    driving_list = np.load('data/scan_epsilon__SinglePhotonDriving/driving_list.npy')[epsilon_index, :n_samples, :, :N] # driving_list here is of shape (n_samples, 2, N).
    theta0 = np.linspace(0, .2, 101)

    n_fock = dim-1
    Gq = np.zeros(n_samples)
    with MPIPoolExecutor() as executor:
        Gc = list(tqdm(executor.map(partial(gc, n_fock, theta0, tau, dim), driving_list), total=n_samples)) # Gc here is of shape (n_samples, n_fock).
        Gq = list(tqdm(executor.map(partial(gc_of_theta0, n_fock=0, theta0=0, tau=tau, dim=dim, only_gq=True), driving_list), total=n_samples)) # Gq is independent of n and theta0
    np.save('data/Gc_measurement.npy', Gc)
    np.save('data/Gq_measurement.npy', Gq)

@timing
def plot_figS1():
    epsilon_dp = [0, 1e-3, 1e-2, 1e-1]
    theta0 = np.linspace(0, .1, 201)
    Gc = np.load('data/Gc_depolarizing.npy')

    data = [np.mean(Gc, axis=1), np.std(Gc, axis=1)]
    fig, axs = plt.subplots(1, 2, layout='constrained', figsize=(figwidth_dc*.67, figwidth_dc*.25))
    legend = [0, '10^{-3}', '10^{-2}', '10^{-1}']
    for i in range(len(epsilon_dp)):
        axs[0].plot(theta0, data[0][i], label=rf'$\epsilon_\mathrm{{dp}}={legend[i]}$', alpha=.6)
        axs[0].fill_between(theta0, data[0][i]-data[1][i], data[0][i]+data[1][i], alpha=.3)
    axs[0].legend(frameon=False, loc='upper right', bbox_to_anchor=[1.05, 1.06], handletextpad=.3, labelspacing=.2, handlelength=1.3)
    axs[0].xaxis.set_major_formatter(FuncFormatter(trim_zeros))
    axs[0].xaxis.set_major_locator(MultipleLocator(.02))
    axs[0].yaxis.set_major_locator(MultipleLocator(5))
    axs[0].minorticks_on()
    axs[0].set_xlim(0, max(theta0))
    axs[0].set_ylim(bottom=0)
    axs[0].set_xlabel(r'$\theta_0$')
    axs[0].set_ylabel(r'$G_c(\theta_0)$')
    axs[0].text(-.21, .95, r'\textbf{a}', transform=axs[0].transAxes)

    Gc = np.load('data/Gc_measurement.npy')
    Gq = np.load('data/Gq_measurement.npy')
    n_samples, n_fock = Gc.shape
    for i in range(n_samples):
        axs[1].plot(np.arange(n_fock), Gc[i]/Gq[i])
    axs[1].minorticks_on()
    axs[1].set_xlim(0, n_fock)
    axs[1].set_xlabel('$n$')
    axs[1].set_ylabel(r'$I_\mathrm{c,max}/I_\mathrm{q}$')
    axs[1].text(-.27, .95, r'\textbf{b}', transform=axs[1].transAxes)

    fig.savefig('figures/figS1.pdf', bbox_inches='tight')


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--depolarizing_noise', action='store_true')
    parser.add_argument('--measurement', action='store_true')
    parser.add_argument('--plot_figS1', action='store_true')
    args = parser.parse_args()

    if args.depolarizing_noise:
        depolarizing_noise()

    if args.measurement:
        measurement()

    if args.plot_figS1:
        plot_figS1()