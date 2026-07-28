from pathlib import Path
import warnings
from argparse import ArgumentParser

import numpy as np
from qutip import expect, position, momentum
from mpi4py.futures import MPIPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator

from utils import timing, mpl_rcParams, figwidth_dc
mpl.rcParams.update(mpl_rcParams)

Path('data').mkdir(exist_ok=True)
Path('figures').mkdir(exist_ok=True)
#------------------------------------------------------------------------------------------------------------------------------------#

def cov(psi0): # compute covariance matrix of psi0
    dim = psi0.shape[0]
    x = position(dim)
    p = momentum(dim)
    sigma = np.zeros((2, 2))
    sigma[0, 0] = (expect(x**2, psi0) - expect(x, psi0)**2).real
    sigma[0, 1] = (expect(x*p+p*x, psi0)/2 - expect(x, psi0)*expect(p, psi0)).real
    sigma[1, 0] = sigma[0, 1]
    sigma[1, 1] = (expect(p**2, psi0) - expect(p, psi0)**2).real
    return sigma

def nonGaussian(psi0):
    sigma = 2*cov(psi0) # this is twice the covariance matrix, see Eq. (3.49) in Ref. Serafini, A. Quantum Continuous Variables: A Primer of Theoretical Methods. (CRC Press, Boca Raton, 2023).
    niu = np.sqrt(np.linalg.det(sigma))
    if niu>1:
        delta = (niu+1)/2*np.log2((niu+1)/2) - (niu-1)/2*np.log2((niu-1)/2) # see Eq. (3.99) in Ref. Serafini, A. Quantum Continuous Variables: A Primer of Theoretical Methods. (CRC Press, Boca Raton, 2023).
    elif np.isclose(niu, 1): # avoid invalid value in log
        delta = 0
    else: # niu<1 is not allowed, see Eq. (3.85) in Ref. Serafini, A. Quantum Continuous Variables: A Primer of Theoretical Methods. (CRC Press, Boca Raton, 2023).
        delta = 0
        warnings.warn(f'{niu=}<1 is not physical.')
    return delta

def subPlanckScale(psi0):
    dim = psi0.shape[0]
    x = position(dim)
    p = momentum(dim)
    var_x = (expect(x**2, psi0) - expect(x, psi0)**2).real
    var_p = (expect(p**2, psi0) - expect(p, psi0)**2).real
    Delta_x = np.sqrt(var_x)
    Delta_p = np.sqrt(var_p)
    scale = 1/(Delta_x * Delta_p) # in natural unit \hbar = 1.
    return scale

@timing
def compute(task):
    n_samples = 1000
    epsilon = np.logspace(1, 3, 21)
    N = 70
    
    result = []
    with MPIPoolExecutor() as executor:
        match task:
            case 'subPlanckScale':
                for i in tqdm(range(len(epsilon)), desc='subPlanckScale'):
                    psi0 = np.load(f'data/scan_epsilon__SinglePhotonDriving/psi0__epsilon{i}.npy', allow_pickle=True) # psi0 is of shape (n_samples, N).
                    temp = list(executor.map(subPlanckScale, psi0.reshape(-1)))
                    temp = np.array(temp).reshape((n_samples, N))
                    result.append(temp)
                np.save('data/subPlanckScale.npy', result) # result is of shape (len(epsilon), n_samples, N).
            
            case 'nonGaussian':
                for i in tqdm(range(len(epsilon)), desc='nonGaussian'):
                    psi0 = np.load(f'data/scan_epsilon__SinglePhotonDriving/psi0__epsilon{i}.npy', allow_pickle=True) # psi0 is of shape (n_samples, N).
                    temp = list(executor.map(nonGaussian, psi0.reshape(-1)))
                    temp = np.array(temp).reshape((n_samples, N))
                    result.append(temp)
                np.save('data/nonGaussian.npy', result) # result is of shape (len(epsilon), n_samples, N).

@timing
def plot_figS4():
    epsilon = np.logspace(1, 3, 21)
    tau = .1
    N_stop = 30
    epsilon_start = 6
    result = [np.load('data/subPlanckScale.npy')[epsilon_start:, :, :N_stop], np.load('data/nonGaussian.npy')[epsilon_start:, :, :N_stop]]

    data = [np.mean(A, axis=1) for A in result]
    norm = ['log', None]
    text = [r'\textbf{a}', r'\textbf{b}']
    fig, axs = plt.subplots(1, 2, layout='constrained', figsize=(figwidth_dc*.67, figwidth_dc*.28))
    for i in range(2):
        cm = axs[i].pcolormesh(np.arange(1, N_stop+1)*tau, epsilon[epsilon_start:], data[i], cmap='rainbow', norm=norm[i])
        cbar = fig.colorbar(cm, ax=axs[i], pad=-.12, location='top', shrink=.68, anchor=(1, 1))
        cbar.ax.tick_params(which='major', length=1.5, pad=0)
        cbar.ax.tick_params(which='minor', length=1)
        cbar.ax.minorticks_on()
        if i==1:
            cbar.ax.xaxis.set_major_locator(MultipleLocator(2))
        axs[i].xaxis.set_major_locator(MultipleLocator(1))
        axs[i].minorticks_on()
        axs[i].set_yscale('log')
        axs[i].set_xlabel(r'$\chi T$')
        axs[i].set_ylabel(r'$\epsilon/\chi$')
        axs[i].text(-.25, 1.05, text[i], transform=axs[i].transAxes)
    
    axs[0].text(.25, 1.025, '$a$', transform=axs[0].transAxes)
    axs[1].text(.25, 1.025, r'$\delta$', transform=axs[1].transAxes)
    CS = axs[0].contour(np.arange(1, N_stop+1)*tau, epsilon[epsilon_start:], data[0], levels=[.006, .01, .02, .04], colors='k', linewidths=.5)
    axs[0].clabel(CS, manual=[(2.5, 800), (2.1, 300), (1.8, 150), (1.6, 70)], fontsize=9, fmt=lambda x: f"{x:g}")
    CS = axs[1].contour(np.arange(1, N_stop+1)*tau, epsilon[epsilon_start:], data[1], levels=[6, 7, 8, 9], colors='k', linewidths=.5)
    axs[1].clabel(CS, manual=[(1.7, 40), (2, 100), (2.3, 300), (2.6, 800)], fontsize=9)
    
    fig.savefig('figures/figS4.pdf')

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--task', choices=['subPlanckScale', 'nonGaussian'])
    parser.add_argument('--plot_figS4', action='store_true')
    args = parser.parse_args()
    
    if args.task is not None:
        compute(args.task)

    if args.plot_figS4:
        plot_figS4()
