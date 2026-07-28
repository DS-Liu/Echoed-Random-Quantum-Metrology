import traceback
from collections.abc import Callable
from datetime import datetime
from functools import wraps
import warnings

import numpy as np
from qutip import destroy, basis, expect, qeye


class Params_H:
    def __init__(self, driving_type, driving_list, tau, dim):
        self.driving_type = driving_type
        self.driving_list = driving_list
        self.tau = tau
        self.dim = dim

def gain(chi, T, n_time_steps, theta, dim=1000, epsilon=0): # metrological gain for quantum Fisher information and classical Fisher information
    a = destroy(dim)

    rng = np.random.default_rng()
    u1_list = rng.uniform(-1, 1, size=n_time_steps)
    u2_list = rng.uniform(-1, 1, size=n_time_steps)
    step_size = T / n_time_steps

    U = qeye(dim)
    for i in range(n_time_steps):
        H = chi*(a.dag()*a)**2 + u1_list[i]*(a.dag()+a) + 1j*u2_list[i]*(a.dag()-a)
        U = (-1j*H*step_size).expm() * U
    psi0 = U * basis(dim, 0)

    # Compute metrological gain for quantum Fisher information.
    num_op = a.dag() * a
    n_ev = expect(num_op, psi0)
    n2_ev = expect(num_op**2, psi0)
    n_var = n2_ev - n_ev**2
    threshold = n_ev + 3*np.sqrt(n_var)
    if threshold > dim:
        warnings.warn(f"Truncation may be inaccurate for chi={chi}, T={T}. Try a truncation number larger than {int(threshold)}.")
    g2 = (n2_ev-n_ev) / n_ev**2                                 # equal-time second-order correlation function g^{(2)}(0), used to indicate photon blockade for large chi.
    Gq = n_var / n_ev                                           # metrological gain for quantum Fisher information.

    # Compute metroligical gain for classical Fisher information.
    psi_theta = (-1j*theta*num_op).expm() * psi0
    psi_f = U.dag() * psi_theta
    amp = psi_f.data_as('ndarray')                              # amplitude of psi_f.
    p = (amp * amp.conjugate()).real.flatten()
    A = U.dag() * (-1j*num_op) * psi_theta * psi_f.dag()
    dp = (A + A.dag()).diag()                                   # derivative of p with respect to theta.
    if epsilon!=0:                                              # depolarizing noise with strength epsilon.
        p = (1-epsilon)*p + epsilon/dim
        dp = (1-epsilon)*dp

    CFI = 0
    for i in range(len(p)):
        if p[i] !=0:
            CFI += dp[i]**2 / p[i]
    Gc = CFI / (4*n_ev)                                         # metrological gain for classical Fisher information.
    return [g2, n_ev, Gq, Gc, p[0], dp[0]]

def Fisher_SPD(chi, theta0, n_time_steps, dim=1000, epsilon=1e-3, tau=1): # FIsher information for single photon detection.
    '''
    Args:
        epsilon: float, strength of depolarizing noise.
        tau: float, step size for time evolution.
    '''
    rng = np.random.default_rng()
    u1_list = rng.uniform(-1, 1, size=n_time_steps)
    u2_list = rng.uniform(-1, 1, size=n_time_steps)

    a = destroy(dim)
    U = qeye(dim)
    for i in range(n_time_steps):
        H = chi*(a.dag()*a)**2 + u1_list[i]*(a.dag()+a) + 1j*u2_list[i]*(a.dag()-a)
        U = (-1j*H*tau).expm() * U
    psi0 = U * basis(dim, 0)

    num_op = a.dag() * a
    n_ev = expect(num_op, psi0)
    n2_ev = expect(num_op**2, psi0)
    n_var = n2_ev - n_ev**2
    threshold = n_ev + 3*np.sqrt(n_var)
    if threshold > dim:
        warnings.warn(f"Truncation may be inaccurate for chi={chi}, n_time_steps={n_time_steps}. Try a truncation number larger than {int(threshold)}.")

    psi0 = psi0.data_as('ndarray').flatten()
    n = np.arange(dim)
    expn = np.exp(-1j*theta0*n)
    psi_theta = expn * psi0
    temp1 = psi0.conjugate() @ psi_theta
    p0 = temp1*temp1.conjugate()
    temp2 = psi0.conjugate()@(-1j*n*psi_theta) * temp1.conjugate()
    dp0 = temp2 + temp2.conjugate()                                   # derivative of p0 with respect to theta.
    if epsilon!=0:                                              # depolarizing noise with strength epsilon.
        p0 = (1-epsilon)*p0 + epsilon/dim
        dp0 = (1-epsilon)*dp0

    CFI = 0
    temp = dp0**2
    if p0 != 0:
        CFI += temp / p0
    if p0 != 1:
        CFI += temp / (1-p0)
    Gc = CFI / (4*n_ev)                                         # metrological gain for classical Fisher information.
    return [n_ev, p0, dp0, CFI, Gc]


###################################################################################################
########################################## For timing #############################################
###################################################################################################
def timing(main: Callable) -> Callable:
    @wraps(main)
    def inner(*args, **kwargs):
        start_time = datetime.now()
        try:
            main(*args, **kwargs)
        except Exception as e:
            print(traceback.format_exc())
        end_time = datetime.now()
        print(f"--- Started at {start_time}, ended at {end_time}, the total execution time is {end_time-start_time} ---")
    return inner


###################################################################################################
######################################### For plotting ############################################
###################################################################################################
mpl_rcParams = {
    'font.size': 10,               # match PRL body text (~10pt LaTeX)
    'axes.labelsize': 10,
    'axes.labelpad': 2,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.fontsize': 10,
    'axes.linewidth': 0.8,        # thinner axes for compact plots
    'lines.linewidth': 1.0,
    'xtick.major.pad': 2.5,
    'ytick.major.pad': 2.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'mathtext.fontset': 'cm',     # Computer Modern for LaTeX match
    'text.usetex': True,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.transparent': True
}
figwidth_sc = 3.375 # figure width for single-column papers
figwidth_dc = 7     # figure width for double-column papers
def trim_zeros(x, pos):
    return ('%g' % x)