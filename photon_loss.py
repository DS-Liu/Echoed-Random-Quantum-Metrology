from pathlib import Path
from argparse import ArgumentParser
from functools import partial
import warnings 

import numpy as np
from scipy.optimize import minimize_scalar, curve_fit
from qutip import destroy, num, expect, mesolve, QobjEvo, basis, fock_dm
from mpi4py.futures import MPIPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

from fig2_statistics import Gq_Gc_thetab_QFI_CFI, compute_Gc_vs_theta0, model
from utils import Params_H, timing, mpl_rcParams, figwidth_dc, trim_zeros
mpl.rcParams.update(mpl_rcParams)

Path('data/photon_loss').mkdir(parents=True, exist_ok=True)
Path('figures').mkdir(exist_ok=True)
#------------------------------------------------------------------------------------------------------------------------------------#

# tau = .1
n_samples = 1000
kappa_list = [0, .001, .002, 0.003, .004]

def Hamiltonian(t, args):
    params_H = args['params_H']
    direction = args['direction']

    u1_list, u2_list = params_H.driving_list
    a = destroy(params_H.dim)
    num_op = num(params_H.dim)
    H0 = num_op**2
    match params_H.driving_type:
        case 'SinglePhotonDriving':
            H1 = a.dag() + a
            H2 = a.dag() - a
        case 'TwoPhotonDriving':
            H1 = a.dag()**2 + a**2
            H2 = a.dag()**2 - a**2

    match direction:
        case 'forward':
            index = min(int(t/params_H.tau), len(u1_list)-1)
            H = H0 + u1_list[index]*H1 + 1j*u2_list[index]*H2
        case 'backward':
            u1_list = u1_list[::-1]
            u2_list = u2_list[::-1]
            index = min(int(t/params_H.tau), len(u1_list)-1)
            H = - H0 - u1_list[index]*H1 - 1j*u2_list[index]*H2
    return H

def qfi(rho0):
    dim = rho0.shape[0]
    num_op = num(dim)

    if rho0.isket: # Quantum Fisher information for pure states.
        QFI = 4*(expect(num_op**2, rho0) - expect(num_op, rho0)**2)
    elif rho0.isoper: # Quantum Fisher information for mixed states.
        eigvals, eigvecs = rho0.eigenstates()
        if eigvals.min() < -1e-5:
            warnings.warn(f'The density matrix is not positive semi-definite, with smallest eigenvalue {eigvals.min():.2e}.')

        QFI = 0
        for i in range(dim):
            for j in range(i+1, dim):
                plus = (eigvals[i] + eigvals[j]).real
                if plus>0:
                    minus = (eigvals[i] - eigvals[j]).real
                    QFI += minus**2 / plus * np.abs(num_op.matrix_element(eigvecs[i], eigvecs[j]))**2
        QFI *= 4
    return QFI

def forward(kappa, params_H):
    N = params_H.driving_list.shape[1]
    a = destroy(params_H.dim)
    num_op = num(params_H.dim)
    c_ops = None if kappa==0 else np.sqrt(kappa)*a

    H = QobjEvo(Hamiltonian, args={'params_H':params_H, 'direction':'forward'})
    result = mesolve(H, basis(params_H.dim, 0), tlist=np.arange(N+1)*params_H.tau, c_ops=c_ops, e_ops=[num_op, num_op**2], options={'store_final_state':True, 'nsteps':int(1e8), 'method':'bdf', 'atol':1e-9, 'rtol':1e-9})
    rho0 = result.final_state
    n_ev = result.expect[0][-1]
    n2_ev = result.expect[1][-1]
    n_std = np.sqrt(n2_ev - n_ev**2)
    threshold = n_ev + 3*n_std
    if threshold>params_H.dim:
        warnings.warn(f"Truncation may be inaccurate. Try a truncation number larger than {int(threshold)}.")

    QFI = qfi(rho0)
    Gq = QFI / (4*n_ev)
    return [n_ev, Gq, rho0]

def cfi(kappa, params_H, rho0, theta0, epsilon_dp=1e-3):
    N = params_H.driving_list.shape[1]
    a = destroy(params_H.dim)
    num_op = num(params_H.dim)
    M0 = fock_dm(params_H.dim, 0)
    M1 = fock_dm(params_H.dim, 1)
    c_ops = None if kappa==0 else np.sqrt(kappa)*a

    U_theta = (-1j*theta0*num_op).expm()
    rho_theta = U_theta * rho0 * U_theta.dag()
    H = QobjEvo(Hamiltonian, args={'params_H':params_H, 'direction':'backward'})
    result = mesolve(H, rho_theta, tlist=np.arange(N+1)*params_H.tau, c_ops=c_ops, e_ops=[M0, M1], options={'nsteps':int(1e8), 'method':'bdf', 'atol':1e-9, 'rtol':1e-9})
    p0 = result.expect[0][-1].real
    p1 = result.expect[1][-1].real
    
    drho_theta = 1j * (rho_theta*num_op - num_op*rho_theta)
    result = mesolve(H, drho_theta, tlist=np.arange(N+1)*params_H.tau, c_ops=c_ops, e_ops=[M0, M1], options={'nsteps':int(1e8), 'method':'bdf', 'atol':1e-9, 'rtol':1e-9})
    dp0 = result.expect[0][-1].real
    dp1 = result.expect[1][-1].real
    
    if epsilon_dp!=0:
        p0 = (1-epsilon_dp)*p0 + epsilon_dp/params_H.dim
        p1 = (1-epsilon_dp)*p1 + epsilon_dp/params_H.dim
        dp0 = (1-epsilon_dp)*dp0
        dp1 = (1-epsilon_dp)*dp1

    p2 = 1 - p0 - p1
    dp2 = - dp0 - dp1
    p_list = [p0, p1, p2]
    dp_list = [dp0, dp1, dp2]

    CFI = 0
    for i in range(3):
        if p_list[i] > 0:
            CFI += dp_list[i]**2 / p_list[i]
    return CFI

def interval(psi0):
    theta0 = np.linspace(0, .8, 801)
    Gc = compute_Gc_vs_theta0(psi0, theta0)
    best_index = np.argmax(Gc)
    b = theta0[-1]

    for i in range(best_index, len(theta0)-1):
        if Gc[i]<Gc[i+1]:
            b = theta0[i]
            break
    
    if b==theta0[-1]:
        warnings.warn('Try increase theta0.')
    assert b>.0001
    return [.0001, b]

def CFI_thetab(kappa, params_H, rho0, bounds, epsilon_dp=1e-3):
    def objective(theta0):
        return -cfi(kappa, params_H, rho0, theta0, epsilon_dp)
    
    res = minimize_scalar(objective, bounds=bounds, method='bounded', options={'maxiter':15, 'xatol':.003}) # xatol is the absolute error for theta0.
    thetab = res.x
    CFI = -res.fun
    
    if not res.success:
        warnings.warn(res.message)
    return [CFI, thetab]


@timing
def get_rho0(dim, N, tau, driving_type, epsilon_index, kappa_index):
    driving_list = np.load(f'data/scan_epsilon__{driving_type}/driving_list.npy')[epsilon_index, :, :, :N] # driving list here is of shape (n_samples, 2, N)

    n_ev = np.zeros(n_samples)
    Gq = np.zeros(n_samples)
    rho0 = np.empty(n_samples, dtype=object)
    with MPIPoolExecutor() as executor:
        params_H_list = [Params_H(driving_type, dl, tau, dim) for dl in driving_list]
        result = executor.map(partial(forward, kappa_list[kappa_index]), params_H_list)
        for i, res in enumerate(tqdm(result, total=n_samples, desc=f'forward, {driving_type}')):
            n_ev[i], Gq[i], rho0[i] = res
        np.save(f'data/photon_loss/n_ev__{driving_type}__epsilon{epsilon_index}__kappa{kappa_index}.npy', n_ev)
        np.save(f'data/photon_loss/Gq__{driving_type}__epsilon{epsilon_index}__kappa{kappa_index}.npy', Gq)
        np.save(f'data/photon_loss/rho0__{driving_type}__epsilon{epsilon_index}__kappa{kappa_index}.npy', rho0)

        if kappa_index==0:
            result = list(tqdm(executor.map(Gq_Gc_thetab_QFI_CFI, rho0), total=n_samples, desc='Gq_Gc_thetab_QFI_CFI for kappa_index==0'))
            np.save(f'data/photon_loss/Gq_Gc_thetab_QFI_CFI__{driving_type}__epsilon{epsilon_index}__kappa0.npy', result) # result is of shape (n_samples, 5).

@timing
def get_CFI_thetab(dim, N, tau, driving_type, epsilon_index, kappa_index):
    driving_list = np.load(f'data/scan_epsilon__{driving_type}/driving_list.npy')[epsilon_index, :, :, :N] # driving_list here is of shape (n_samples, 2, N).
    psi0 = np.load(f'data/photon_loss/rho0__{driving_type}__epsilon{epsilon_index}__kappa0.npy', allow_pickle=True) # psi0 here is of shape (n_samples,).
    rho0 = np.load(f'data/photon_loss/rho0__{driving_type}__epsilon{epsilon_index}__kappa{kappa_index}.npy', allow_pickle=True) # rho0 here is of shape (n_samples,).
    n_ev = np.load(f'data/photon_loss/n_ev__{driving_type}__epsilon{epsilon_index}__kappa{kappa_index}.npy') # n_ev here is of shape (n_samples,).
    bounds = [interval(psi0) for psi0 in psi0]

    CFI = np.zeros(n_samples)
    thetab = np.zeros(n_samples)
    with MPIPoolExecutor() as executor:
        params_H_list = [Params_H(driving_type, dl, tau, dim) for dl in driving_list]
        result = executor.map(partial(CFI_thetab, kappa_list[kappa_index]), params_H_list, rho0, bounds)
        for i, res in enumerate(tqdm(result, total=n_samples, desc=f'get_CFI_thetab, {driving_type}, kappa{kappa_index}')):
            CFI[i], thetab[i] = res
        
        Gc = CFI / (4*n_ev)
        np.save(f'data/photon_loss/Gc_CFI_thetab__{driving_type}__epsilon{epsilon_index}__kappa{kappa_index}.npy', [Gc, CFI, thetab])

@timing
def get_Gc_vs_theta0(dim, N, tau, driving_type, epsilon_index, sample_index):
    driving_list = np.load(f'data/scan_epsilon__{driving_type}/driving_list.npy')[epsilon_index, sample_index, :, :N]
    params_H = Params_H(driving_type, driving_list, tau, dim)

    theta0 = np.linspace(0, .2, 101)
    Gc = []
    with MPIPoolExecutor() as executor:
        for i in tqdm([1, 2, 3, 4], desc=f'get_Gc_vs_theta0, {driving_type}, sample{sample_index}'):
            rho0 = np.load(f'data/photon_loss/rho0__{driving_type}__epsilon{epsilon_index}__kappa{i}.npy', allow_pickle=True)[sample_index]
            n_ev = np.load(f'data/photon_loss/n_ev__{driving_type}__epsilon{epsilon_index}__kappa{i}.npy')[sample_index]

            CFI = np.array(list(executor.map(partial(cfi, kappa_list[i], params_H, rho0), theta0)))
            Gc.append(CFI / (4*n_ev))
        np.save(f'data/photon_loss/Gc_vs_theta0__{driving_type}__epsilon{epsilon_index}__sample{sample_index}.npy', Gc) # Gc is of shape (4, len(theta0)).
    

@timing
def plot_fig3():
    fig, axs = plt.subplots(1, 3, layout='constrained', figsize=(figwidth_dc, figwidth_dc*.28))

    # Control fluctuations
    Gc_random = np.load('data/control_error/Gc_SinglePhotonDriving_Randomized_dim210_epsilon40.0_N75_tau0.02_clipFalse.npy')    
    Delta_epsilon = np.linspace(0, .03, 51)
    axs[0].plot(Delta_epsilon, Gc_random.mean(axis=0))
    axs[0].fill_between(Delta_epsilon, Gc_random.mean(axis=0)-Gc_random.std(axis=0), Gc_random.mean(axis=0)+Gc_random.std(axis=0), alpha=.7)

    Gc_random = np.load('data/control_error/Gc_TwoPhotonDriving_Randomized_dim210_epsilon6.0_N40_tau0.02_clipFalse.npy')
    Delta_epsilon = np.linspace(0, .03, 51)
    axs[0].plot(Delta_epsilon, Gc_random.mean(axis=0))
    axs[0].fill_between(Delta_epsilon, Gc_random.mean(axis=0)-Gc_random.std(axis=0), Gc_random.mean(axis=0)+Gc_random.std(axis=0), alpha=.7)

    axs[0].set_xlim(0, .03)
    axs[0].set_ylim(12, 18)
    axs[0].set_xlabel(r'$\Delta\epsilon/\epsilon$')
    axs[0].set_ylabel(r'$G_\mathrm{c,max}$')
    axs[0].minorticks_on()
    axs[0].xaxis.set_major_locator(MultipleLocator(.01))
    axs[0].yaxis.set_major_locator(MultipleLocator(1))
    axs[0].xaxis.set_major_formatter(FuncFormatter(trim_zeros))
    axs[0].text(-.22, 1, r'\textbf{a}', transform=axs[0].transAxes)


    # Single-photon driving, photon loss
    nbar_SPD_kappa0 = np.hstack([np.load(f'data/photon_loss/n_ev__SinglePhotonDriving__epsilon{i}__kappa0.npy') for i in [6, 10]])
    nbar_SPD_kappa1 = np.hstack([np.load(f'data/photon_loss/n_ev__SinglePhotonDriving__epsilon{i}__kappa1.npy') for i in [6, 10]])
    nbar_SPD_kappa2 = np.hstack([np.load(f'data/photon_loss/n_ev__SinglePhotonDriving__epsilon{i}__kappa2.npy') for i in [6, 10]])
    nbar_SPD_kappa3 = np.hstack([np.load(f'data/photon_loss/n_ev__SinglePhotonDriving__epsilon{i}__kappa3.npy') for i in [6, 10]])
    nbar_SPD_kappa4 = np.hstack([np.load(f'data/photon_loss/n_ev__SinglePhotonDriving__epsilon{i}__kappa4.npy') for i in [6, 10]])
    CFI_SPD_kappa0 = np.hstack([np.load(f'data/photon_loss/Gq_Gc_thetab_QFI_CFI__SinglePhotonDriving__epsilon{i}__kappa0.npy')[:, 4] for i in [6, 10]])
    CFI_SPD_kappa1 = np.hstack([np.load(f'data/photon_loss/Gc_CFI_thetab__SinglePhotonDriving__epsilon{i}__kappa1.npy')[1] for i in [6, 10]])
    CFI_SPD_kappa2 = np.hstack([np.load(f'data/photon_loss/Gc_CFI_thetab__SinglePhotonDriving__epsilon{i}__kappa2.npy')[1] for i in [6, 10]])
    CFI_SPD_kappa3 = np.hstack([np.load(f'data/photon_loss/Gc_CFI_thetab__SinglePhotonDriving__epsilon{i}__kappa3.npy')[1] for i in [6, 10]])
    CFI_SPD_kappa4 = np.hstack([np.load(f'data/photon_loss/Gc_CFI_thetab__SinglePhotonDriving__epsilon{i}__kappa4.npy')[1] for i in [6, 10]])
    nbar_SPD = [nbar_SPD_kappa0, nbar_SPD_kappa1, nbar_SPD_kappa2, nbar_SPD_kappa3, nbar_SPD_kappa4]
    CFI_SPD = [CFI_SPD_kappa0, CFI_SPD_kappa1, CFI_SPD_kappa2, CFI_SPD_kappa3, CFI_SPD_kappa4]

    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=min(kappa_list), vmax=max(kappa_list))
    colors = cmap(norm(kappa_list))
    b_list = np.zeros(len(kappa_list))
    for i in range(5):
        axs[1].scatter(nbar_SPD[i], CFI_SPD[i], s=1, color=colors[i], edgecolor='none', alpha=.8)
        
        popt, _ = curve_fit(model, nbar_SPD[i], CFI_SPD[i], p0=[2, 1.5])
        a, b = popt
        b_list[i] = b
        print(f'optimal parameters for Ic, kappa{i}: {a=:.2f}, {b=:.2f}.')
        axs[1].plot(np.arange(1, int(max(nbar_SPD[i]))), model(np.arange(1, int(max(nbar_SPD[i]))), *popt), color=colors[i], linestyle='--')
    axs[1].plot(np.arange(int(np.max(nbar_SPD[0]))), 4*np.arange(int(np.max(nbar_SPD[0]))), 'k--')
    axs[1].set_yticks([0, 1000, 2000, 3000, 4000])
    axs[1].set_yticklabels(['0', '1', '2', '3', '4'])
    axs[1].text(-.18, .85, r'$\times 10^3$', transform=axs[1].transAxes)
    axs[1].xaxis.set_major_locator(MultipleLocator(10))
    axs[1].minorticks_on()
    axs[1].set_xlim(0, np.max(nbar_SPD[0])+1)
    axs[1].set_ylim(0, np.max(CFI_SPD[0])+.5)
    axs[1].set_xlabel(r'$\langle\hat{n}\rangle$')
    axs[1].set_ylabel(r'$I_\mathrm{c,max}$')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axs[1], pad=-.1, location='top', shrink=.68, anchor=(1, 1))
    cbar.ax.tick_params(which='major', length=2, pad=.5)
    cbar.ax.tick_params(which='minor', length=1.5)
    cbar.ax.xaxis.set_major_formatter(FuncFormatter(trim_zeros))
    cbar.ax.xaxis.set_major_locator(MultipleLocator(.002))
    cbar.ax.xaxis.set_minor_locator(MultipleLocator(.001))
    axs[1].text(.1, 1.03, r'$\kappa/\chi$', transform=axs[1].transAxes)
    axs[1].text(-.2, 1, r'\textbf{b}', transform=axs[1].transAxes)

    ax_inset = axs[1].inset_axes([.04, .57, .37, .38])
    ax_inset.plot(kappa_list, b_list, marker='*', markersize=4)
    ax_inset.set_xlim(-.0002, .0042)
    ax_inset.set_ylim(1.35, 1.8)
    ax_inset.set_xlabel(r'$\kappa/\chi$', fontsize=9, labelpad=.5)
    ax_inset.yaxis.tick_right()
    ax_inset.yaxis.set_label_position("right")
    ax_inset.minorticks_on()
    ax_inset.tick_params(which='major', labelsize=8, length=2.5, width=.7, pad=.5) 
    ax_inset.tick_params(which='minor', length=1.5, width=.5)
    ax_inset.xaxis.set_minor_locator(MultipleLocator(.001))
    ax_inset.yaxis.set_major_locator(MultipleLocator(.2))
    ax_inset.set_xticks([0, .002], labels=['$0$', '$0.002$'])


    # TwoPhotonDriving, photon loss
    nbar_TPD_kappa0 = np.hstack([np.load(f'data/photon_loss/n_ev__TwoPhotonDriving__epsilon{i}__kappa0.npy') for i in [10, 13]])
    nbar_TPD_kappa1 = np.hstack([np.load(f'data/photon_loss/n_ev__TwoPhotonDriving__epsilon{i}__kappa1.npy') for i in [10, 13]])
    nbar_TPD_kappa2 = np.hstack([np.load(f'data/photon_loss/n_ev__TwoPhotonDriving__epsilon{i}__kappa2.npy') for i in [10, 13]])
    nbar_TPD_kappa3 = np.hstack([np.load(f'data/photon_loss/n_ev__TwoPhotonDriving__epsilon{i}__kappa3.npy') for i in [10, 13]])
    nbar_TPD_kappa4 = np.hstack([np.load(f'data/photon_loss/n_ev__TwoPhotonDriving__epsilon{i}__kappa4.npy') for i in [10, 13]])
    CFI_TPD_kappa0 = np.hstack([np.load(f'data/photon_loss/Gq_Gc_thetab_QFI_CFI__TwoPhotonDriving__epsilon{i}__kappa0.npy')[:, 4] for i in [10, 13]])
    CFI_TPD_kappa1 = np.hstack([np.load(f'data/photon_loss/Gc_CFI_thetab__TwoPhotonDriving__epsilon{i}__kappa1.npy')[1] for i in [10, 13]])
    CFI_TPD_kappa2 = np.hstack([np.load(f'data/photon_loss/Gc_CFI_thetab__TwoPhotonDriving__epsilon{i}__kappa2.npy')[1] for i in [10, 13]])
    CFI_TPD_kappa3 = np.hstack([np.load(f'data/photon_loss/Gc_CFI_thetab__TwoPhotonDriving__epsilon{i}__kappa3.npy')[1] for i in [10, 13]])
    CFI_TPD_kappa4 = np.hstack([np.load(f'data/photon_loss/Gc_CFI_thetab__TwoPhotonDriving__epsilon{i}__kappa4.npy')[1] for i in [10, 13]])
    nbar_TPD = [nbar_TPD_kappa0, nbar_TPD_kappa1, nbar_TPD_kappa2, nbar_TPD_kappa3, nbar_TPD_kappa4]
    CFI_TPD = [CFI_TPD_kappa0, CFI_TPD_kappa1, CFI_TPD_kappa2, CFI_TPD_kappa3, CFI_TPD_kappa4]

    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=min(kappa_list), vmax=max(kappa_list))
    colors = cmap(norm(kappa_list))
    b_list = np.zeros(len(kappa_list))
    for i in range(5):
        axs[2].scatter(nbar_TPD[i], CFI_TPD[i], s=1, color=colors[i], edgecolor='none', alpha=.8)
        
        popt, _ = curve_fit(model, nbar_TPD[i], CFI_TPD[i], p0=[2, 1.5])
        a, b = popt
        b_list[i] = b
        print(f'optimal parameters for Ic, kappa{i}: {a=:.2f}, {b=:.2f}.')
        axs[2].plot(np.arange(1, int(max(nbar_TPD[i]))), model(np.arange(1, int(max(nbar_TPD[i]))), *popt), color=colors[i], linestyle='--')
    axs[2].plot(np.arange(int(np.max(nbar_TPD[0]))), 4*np.arange(int(np.max(nbar_TPD[0]))), 'k--')
    axs[2].set_yticks([0, 2000, 4000, 6000])
    axs[2].set_yticklabels(['0', '2', '4', '6'])
    axs[2].text(-.18, .85, r'$\times 10^3$', transform=axs[2].transAxes)
    axs[2].xaxis.set_major_locator(MultipleLocator(10))
    axs[2].minorticks_on()
    axs[2].set_xlim(0, np.max(nbar_TPD[0])+1)
    axs[2].set_ylim(0, np.max(CFI_TPD[0])+.5)
    axs[2].set_xlabel(r'$\langle\hat{n}\rangle$')
    axs[2].set_ylabel(r'$I_\mathrm{c,max}$')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axs[2], pad=-.1, location='top', shrink=.68, anchor=(1, 1))
    cbar.ax.tick_params(which='major', length=2, pad=.5)
    cbar.ax.tick_params(which='minor', length=1.5)
    cbar.ax.xaxis.set_major_formatter(FuncFormatter(trim_zeros))
    cbar.ax.xaxis.set_major_locator(MultipleLocator(.002))
    # cbar.ax.set_xticks([0, .002, .004], labels=['$0$', '$0.002$', '$0.004$'])
    cbar.ax.xaxis.set_minor_locator(MultipleLocator(.001))
    axs[2].text(.1, 1.03, r'$\kappa/\chi$', transform=axs[2].transAxes)
    axs[2].text(-.2, 1, r'\textbf{c}', transform=axs[2].transAxes)

    ax_inset = axs[2].inset_axes([.04, .57, .37, .38])
    ax_inset.plot(kappa_list, b_list, marker='*', markersize=4)
    ax_inset.set_xlim(-.0002, .0042)
    ax_inset.set_ylim(1.35, 1.8)
    ax_inset.set_xlabel(r'$\kappa/\chi$', fontsize=9, labelpad=.5)
    ax_inset.yaxis.tick_right()
    ax_inset.yaxis.set_label_position("right")
    ax_inset.minorticks_on()
    ax_inset.tick_params(which='major', labelsize=8, length=2.5, width=.7, pad=.5) 
    ax_inset.tick_params(which='minor', length=1.5, width=.5)
    ax_inset.xaxis.set_minor_locator(MultipleLocator(.001))
    ax_inset.yaxis.set_major_locator(MultipleLocator(.2))
    ax_inset.set_xticks([0, .002], labels=['$0$', '$0.002$'])
    # ax_inset.set_yticks([1.4, 1.6, 1.8], labels=['$1.4$', '$1.6$', '$1.8$'])

    fig.savefig('figures/fig3.pdf', bbox_inches='tight', transparent=True)

def plot_figS7():
    fig, axs = plt.subplots(1, 2, layout='constrained', figsize=(figwidth_dc*.67, figwidth_dc*.28))
    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=min(kappa_list), vmax=max(kappa_list))
    colors = cmap(norm(kappa_list))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    theta0 = np.linspace(0, .2, 101)

    rho0_SPD_epsilon6_kappa0 = np.load('data/photon_loss/rho0__SinglePhotonDriving__epsilon6__kappa0.npy', allow_pickle=True)[0]
    rho0_SPD_epsilon10_kappa0 = np.load('data/photon_loss/rho0__SinglePhotonDriving__epsilon10__kappa0.npy', allow_pickle=True)[0]
    rho0_TPD_epsilon10_kappa0 = np.load('data/photon_loss/rho0__TwoPhotonDriving__epsilon10__kappa0.npy', allow_pickle=True)[0]
    rho0_TPD_epsilon13_kappa0 = np.load('data/photon_loss/rho0__TwoPhotonDriving__epsilon13__kappa0.npy', allow_pickle=True)[0]

    Gc_SPD_epsilon6_kappa0 = compute_Gc_vs_theta0(rho0_SPD_epsilon6_kappa0, theta0)
    Gc_SPD_epsilon10_kappa0 = compute_Gc_vs_theta0(rho0_SPD_epsilon10_kappa0, theta0)
    Gc_TPD_epsilon10_kappa0 = compute_Gc_vs_theta0(rho0_TPD_epsilon10_kappa0, theta0)
    Gc_TPD_epsilon13_kappa0 = compute_Gc_vs_theta0(rho0_TPD_epsilon13_kappa0, theta0)

    axs[0].plot(theta0, Gc_SPD_epsilon6_kappa0, '--', color=colors[0])
    axs[0].plot(theta0, Gc_SPD_epsilon10_kappa0, color=colors[0])
    axs[1].plot(theta0, Gc_TPD_epsilon10_kappa0, '--', color=colors[0])
    axs[1].plot(theta0, Gc_TPD_epsilon13_kappa0, color=colors[0])

    data_SPD_epsilon6 = np.load('data/photon_loss/Gc_vs_theta0__SinglePhotonDriving__epsilon6__sample0.npy')
    data_SPD_epsilon10 = np.load('data/photon_loss/Gc_vs_theta0__SinglePhotonDriving__epsilon10__sample0.npy')
    data_TPD_epsilon10 = np.load('data/photon_loss/Gc_vs_theta0__TwoPhotonDriving__epsilon10__sample0.npy')
    data_TPD_epsilon13 = np.load('data/photon_loss/Gc_vs_theta0__TwoPhotonDriving__epsilon13__sample0.npy')
    for i in range(4):
        axs[0].plot(theta0, data_SPD_epsilon6[i], '--', color=colors[i+1])
        axs[0].plot(theta0, data_SPD_epsilon10[i], color=colors[i+1])
        axs[1].plot(theta0, data_TPD_epsilon10[i], '--', color=colors[i+1])
        axs[1].plot(theta0, data_TPD_epsilon13[i], color=colors[i+1])
    for i in range(2):
        axs[i].hlines(1, theta0.min(), theta0.max(), colors='k', linestyles='dashed')
        axs[i].hlines(.768, 0, 1, colors='r', linestyles='dashed', transform=axs[i].transAxes)
        axs[i].set_xlim(0, .2)
        axs[i].set_xlabel(r'$\theta_0$')
        axs[i].set_ylabel(r'$G_\mathrm{c}(\theta_0)$')
    axs[0].yaxis.set_major_locator(MultipleLocator(2))
    axs[0].set_ylim(0, Gc_SPD_epsilon10_kappa0.max())
    axs[1].set_ylim(0, Gc_TPD_epsilon13_kappa0.max())

    text = [r'\textbf{a}', r'\textbf{b}'] 
    for i in range(2):
        cbar = fig.colorbar(sm, ax=axs[i], pad=-.1, location='top', shrink=.68, anchor=(1, 1))
        cbar.ax.tick_params(which='major', length=1.5, pad=.5)
        cbar.ax.tick_params(which='minor', length=1)
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(trim_zeros))
        cbar.ax.xaxis.set_minor_locator(MultipleLocator(.001))
        axs[i].xaxis.set_major_formatter(FuncFormatter(trim_zeros))
        axs[i].text(.16, 1.03, r'$\kappa/\chi$', transform=axs[i].transAxes)
        axs[i].text(-.2, 1, text[i], transform=axs[i].transAxes)

    fig.savefig('figures/figS7.pdf')


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--get_rho0', action='store_true')
    parser.add_argument('--get_CFI_thetab', action='store_true')
    parser.add_argument('--get_Gc_vs_theta0', action='store_true')

    parser.add_argument('--dim', type=int)
    parser.add_argument('--epsilon_index', type=int)
    parser.add_argument('--N', type=int)
    parser.add_argument('--tau', type=float)
    parser.add_argument('--driving_type', choices=['SinglePhotonDriving', 'TwoPhotonDriving'])
    parser.add_argument('--kappa_index', type=int)
    parser.add_argument('--sample_index', type=int)
    parser.add_argument('--plot_fig3', action='store_true')
    parser.add_argument('--plot_figS7', action='store_true')
    args = parser.parse_args()
    
    if args.get_rho0:
        get_rho0(args.dim, args.N, args.tau, args.driving_type, args.epsilon_index, args.kappa_index)

    if args.get_CFI_thetab:
        get_CFI_thetab(args.dim, args.N, args.tau, args.driving_type, args.epsilon_index, args.kappa_index)

    if args.get_Gc_vs_theta0:
        get_Gc_vs_theta0(args.dim, args.N, args.tau, args.driving_type, args.epsilon_index, args.sample_index)

    if args.plot_fig3:
        plot_fig3()
    
    if args.plot_figS7:
        plot_figS7()