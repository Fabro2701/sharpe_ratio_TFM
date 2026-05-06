"""
Joint sampling distribution of (SR_hat, VaR_hat): empirical vs parametric.

Theoretical asymptotic CIs are computed from the closed-form Omega matrices,
with the empirical version using influence-function partial moments evaluated
numerically against the DGP's cdf. Empirical coverage of those CIs is
estimated by Monte Carlo and reported next to the legend labels.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from scipy.stats import norm, chi2, t
from scipy.optimize import brentq
from scipy.integrate import quad


# =============================================================================
# 1. DGPs
# =============================================================================
class DGP:
    def simulate(self, rng, n):  raise NotImplementedError
    def cdf(self, x):            raise NotImplementedError
    def get_theomoments(self):   raise NotImplementedError


class NormalDGP(DGP):
    def __init__(self, mu, sigma):
        self.mu, self.sigma = float(mu), float(sigma)
    def simulate(self, rng, n):
        return rng.normal(self.mu, self.sigma, n)
    def pdf(self, x):
        return float(norm.pdf(x, self.mu, self.sigma))
    def cdf(self, x):
        return float(norm.cdf(x, self.mu, self.sigma))
    def get_theomoments(self):
        return {'mu': self.mu, 'sigma': self.sigma, 'skew': 0.0, 'kurt': 3.0}


class GaussianMixtureDGP(DGP):
    """X ~ sum_k w_k * N(m_k, s_k^2)."""
    def __init__(self, weights, means, sigmas):
        self.w = np.asarray(weights, dtype=float)
        self.m = np.asarray(means,   dtype=float)
        self.s = np.asarray(sigmas,  dtype=float)
        assert np.isclose(self.w.sum(), 1.0), "weights must sum to 1"
    def simulate(self, rng, n):
        idx = rng.choice(len(self.w), size=n, p=self.w)
        return rng.normal(self.m[idx], self.s[idx])
    def pdf(self, x):
        return float(np.sum(self.w * norm.pdf(x, self.m, self.s)))
    def cdf(self, x):
        return float(np.sum(self.w * norm.cdf(x, self.m, self.s)))
    def get_theomoments(self):
        mu  = float(np.sum(self.w * self.m))
        c   = self.m - mu
        var = float(np.sum(self.w * (self.s**2 + c**2)))
        sigma = np.sqrt(var)
        ex3 = float(np.sum(self.w * (c**3 + 3*c*self.s**2)))
        ex4 = float(np.sum(self.w * (c**4 + 6*c**2*self.s**2 + 3*self.s**4)))
        return {'mu': mu, 'sigma': sigma,
                'skew': ex3/sigma**3, 'kurt': ex4/sigma**4}
    
class StudentTDGP(DGP):
    """X ~ Student-t(nu, loc=mu, scale=sigma)."""
    def __init__(self, mu, sigma, nu):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.nu = float(nu)
        assert self.nu > 0, "degrees of freedom must be positive"
    def simulate(self, rng, n):
        return self.mu + self.sigma * rng.standard_t(self.nu, size=n)
    def pdf(self, x):
        return float(t.pdf(x, df=self.nu, loc=self.mu, scale=self.sigma))
    def cdf(self, x):
        return float(t.cdf(x, df=self.nu, loc=self.mu, scale=self.sigma))
    def get_theomoments(self):
        mu = self.mu if self.nu > 1 else np.nan
        if self.nu > 2:
            var = self.sigma**2 * self.nu / (self.nu - 2)
            sigma = np.sqrt(var)
        else:
            sigma = np.nan
        skew = 0.0 if self.nu > 3 else np.nan
        if self.nu > 4:
            kurt = 3 + 6 / (self.nu - 4)
        else:
            kurt = np.nan
        return {'mu': mu, 'sigma': sigma,
                'skew': skew, 'kurt': kurt}


# =============================================================================
# 2. Computation
# =============================================================================
def _omegas(dgp, alpha):
    """Build theoretical Omega matrices (empirical & parametric-under-normality)."""
    moms       = dgp.get_theomoments()
    mu, sigma  = moms['mu'], moms['sigma']
    g3, g4     = moms['skew'], moms['kurt']
    SR         = mu / sigma
    z_a        = norm.ppf(alpha)

    # True alpha-quantile and density there (pdf via central difference of cdf)
    q = brentq(lambda x: dgp.cdf(x) - alpha, mu - 30*sigma, mu + 30*sigma)
    f_q = dgp.pdf(q)

    # Parametric Omega (formulas under the WORKING ASSUMPTION of normality)
    Op = np.array([[1 + SR**2/2,        mu*z_a/2 - sigma],
                   [mu*z_a/2 - sigma,   sigma**2 * (1 + z_a**2/2)]])

    # Empirical Omega (correct under any F)
    M1, _ = quad(lambda x: (x - mu)    *  dgp.pdf(x), -np.inf, q)
    M2, _ = quad(lambda x: (x - mu)**2 *  dgp.pdf(x), -np.inf, q)
    O11 = 1 + SR**2/2 - SR*g3 + (SR**2/4)*(g4 - 3)
    O22 = alpha*(1 - alpha) / f_q**2
    O12 = (M1/sigma - (SR/(2*sigma**2)) * (M2 - sigma**2 * alpha)) / f_q
    Oe = np.array([[O11, O12], [O12, O22]])

    return {
        'mu': mu, 'sigma': sigma, 'SR_true': SR,
        'q_true': q, 'VaR_true': -q, 'f_at_q': f_q,
        'VaR_par_target': -(mu + sigma * z_a),
        'Omega_emp': Oe, 'Omega_par': Op,
        'z_alpha': z_a, 'moments': moms,
    }


def analyze(dgp, alpha=0.05, T=250, n_sim=5000, tau=0.95, seed=7):
    """
    Theoretical Omegas + Monte Carlo + 2D coverage of the CI ellipse by the truth.

    Returns a dict: 'truth', 'asym_target', 'Omega', 'mc', 'diagnostics',
                    'config', 'moments'.
    """
    rng = np.random.default_rng(seed)
    th  = _omegas(dgp, alpha)
    z_a = th['z_alpha']
    SR_t, VaR_t = th['SR_true'], th['VaR_true']

    # ----- Monte Carlo -----
    SR_h, V_par_h, V_emp_h = (np.empty(n_sim) for _ in range(3))
    for i in range(n_sim):
        X  = dgp.simulate(rng, T)
        mh, sh = X.mean(), X.std(ddof=1)
        SR_h[i]    = mh / sh
        V_par_h[i] = -(mh + sh * z_a)
        V_emp_h[i] = -np.quantile(X, alpha)

    # ----- Empirical 2D coverage of (SR_t, VaR_t) by the theoretical CIs -----
    chi2_thr = chi2.ppf(tau, 2)
    inv_e = np.linalg.inv(th['Omega_emp'] / T)
    inv_p = np.linalg.inv(th['Omega_par'] / T)
    de = np.column_stack([SR_h - SR_t, V_emp_h - VaR_t])
    dp = np.column_stack([SR_h - SR_t, V_par_h - VaR_t])
    cov_e = float(np.mean(np.einsum('ij,jk,ik->i', de, inv_e, de) <= chi2_thr))
    cov_p = float(np.mean(np.einsum('ij,jk,ik->i', dp, inv_p, dp) <= chi2_thr))

    return {
        'truth':       {'SR': SR_t, 'VaR': VaR_t},
        'asym_target': {'SR': SR_t, 'VaR_par': th['VaR_par_target'],
                        'asymp_bias_par': th['VaR_par_target'] - VaR_t},
        'Omega':       {'emp': th['Omega_emp'], 'par': th['Omega_par']},
        'mc':          {'SR': SR_h, 'VaR_emp': V_emp_h, 'VaR_par': V_par_h},
        'diagnostics': {
            'mc_bias_emp':  float(V_emp_h.mean() - VaR_t),
            'mc_bias_par':  float(V_par_h.mean() - VaR_t),
            'mc_var_emp':   float(V_emp_h.var()),
            'mc_var_par':   float(V_par_h.var()),
            'mc_mse_emp':   float(np.mean((V_emp_h - VaR_t)**2)),
            'mc_mse_par':   float(np.mean((V_par_h - VaR_t)**2)),
            'coverage_emp': cov_e,
            'coverage_par': cov_p,
        },
        'config':  {'alpha': alpha, 'T': T, 'n_sim': n_sim, 'tau': tau, 'z_alpha': z_a},
        'moments': th['moments'],
    }


# =============================================================================
# 3. Visualization
# =============================================================================
def _ellipse(mean, cov, conf, **kw):
    eig, vec = np.linalg.eigh(cov)
    order    = eig.argsort()[::-1]
    eig, vec = eig[order], vec[:, order]
    angle    = np.degrees(np.arctan2(vec[1, 0], vec[0, 0]))
    w, h     = 2 * np.sqrt(chi2.ppf(conf, 2) * eig)
    return Ellipse(xy=mean, width=w, height=h, angle=angle, **kw)


def plot_results(results, dgp_label, show_scatter=True, save_path=None, n_scatter=5000):
    """Theoretical ellipses + (optional) MC scatter + marginal Gaussian densities."""
    truth, asym = results['truth'], results['asym_target']
    Om, mc, diag, cfg, moms = (results[k] for k in
                               ('Omega', 'mc', 'diagnostics', 'config', 'moments'))
    SR_t, VaR_t = truth['SR'], truth['VaR']
    VaR_p_tgt   = asym['VaR_par']
    T, tau      = cfg['T'], cfg['tau']
    Sig_e, Sig_p = Om['emp']/T, Om['par']/T

    fig = plt.figure(figsize=(11, 11))
    gs  = gridspec.GridSpec(3, 3, wspace=0.05, hspace=0.05,
                            width_ratios=[3, 3, 1.2], height_ratios=[1.2, 3, 3])
    ax_main  = fig.add_subplot(gs[1:, :2])
    ax_top   = fig.add_subplot(gs[0, :2], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, 2], sharey=ax_main)

    if show_scatter:
        n_total = len(mc['SR'])
        idx = np.random.choice(n_total, size=min(n_scatter, n_total), replace=False)
        ax_main.scatter(mc['SR'][idx], mc['VaR_emp'][idx], s=3, alpha=0.06,
                        color='#1f77b4', rasterized=True)
        ax_main.scatter(mc['SR'][idx], mc['VaR_par'][idx], s=3, alpha=0.06,
                        color='#d62728', rasterized=True)

    # Theoretical ellipses with empirical coverage in label
    cE, cP = diag['coverage_emp'], diag['coverage_par']
    ax_main.add_patch(_ellipse(
        (SR_t, VaR_t), Sig_e, tau,
        facecolor='none', edgecolor='#1f77b4', lw=2.5,
        label=f"Empirical {int(tau*100)}% CI   (coverage = {cE:.1%})"))
    ax_main.add_patch(_ellipse(
        (SR_t, VaR_p_tgt), Sig_p, tau,
        facecolor='none', edgecolor='#d62728', lw=2.5, ls='--',
        label=f"Parametric {int(tau*100)}% CI  (coverage = {cP:.1%})"))

    # Anchors
    ax_main.plot(SR_t, VaR_t, '*', color='black', ms=22, zorder=10,
                 label='True (SR, VaR)')
    ax_main.plot(SR_t, VaR_p_tgt, 's', color='#d62728', mec='black',
                 ms=10, zorder=10, label='Parametric point estimation')

    # Bias arrow (only when there is a visible bias)
    if abs(VaR_p_tgt - VaR_t) > 1e-4:
        ax_main.annotate('', xy=(SR_t, VaR_p_tgt), xytext=(SR_t, VaR_t),
                         arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        ax_main.text(SR_t + 0.005, (VaR_t + VaR_p_tgt)/2,
                     f"bias = {asym['asymp_bias_par']:+.3f}",
                     style='italic', fontsize=10)

    ax_main.set_xlabel(r'$\widehat{SR}$', fontsize=13)
    ax_main.set_ylabel(r'$\widehat{VaR}_{\alpha}$', fontsize=13)
    ax_main.legend(loc='lower left', fontsize=10, framealpha=0.95)
    ax_main.grid(True, ls=':', alpha=0.6)

    # Top: SR marginal (theoretical Gaussian, centered on SR_t for both)
    s_e = np.sqrt(Om['emp'][0, 0]/T); s_p = np.sqrt(Om['par'][0, 0]/T)
    grid = np.linspace(SR_t - 4*max(s_e, s_p), SR_t + 4*max(s_e, s_p), 300)
    # ax_top.plot(grid, norm.pdf(grid, SR_t, s_e), color='#1f77b4', lw=2, label='Empirical')
    # ax_top.plot(grid, norm.pdf(grid, SR_t, s_p), color='#d62728', lw=2, ls='--',
    #             label='Parametric')
    # ax_top.fill_between(grid, norm.pdf(grid, SR_t, s_e), alpha=0.15, color='#1f77b4')
    # ax_top.fill_between(grid, norm.pdf(grid, SR_t, s_p), alpha=0.15, color='#d62728')
    ax_top.plot(grid, norm.pdf(grid, SR_t, s_e), color="#7d0088", lw=2, label='SR (shared)')
    ax_top.fill_between(grid, norm.pdf(grid, SR_t, s_e), alpha=0.15, color="#9c02aa")
    ax_top.axvline(SR_t, color='black', lw=1.2, label='True SR')
    ax_top.set_ylabel('Density', fontsize=10)
    ax_top.legend(loc='upper right', fontsize=9)
    ax_top.grid(True, ls=':', alpha=0.6)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # Right: VaR marginal (empirical centered at truth, parametric at its target)
    v_e = np.sqrt(Om['emp'][1, 1]/T); v_p = np.sqrt(Om['par'][1, 1]/T)
    lo = min(VaR_t, VaR_p_tgt) - 4*max(v_e, v_p)
    hi = max(VaR_t, VaR_p_tgt) + 4*max(v_e, v_p)
    g  = np.linspace(lo, hi, 300)
    ax_right.plot(norm.pdf(g, VaR_t,    v_e), g, color='#1f77b4', lw=2, label='Empirical Var')
    ax_right.plot(norm.pdf(g, VaR_p_tgt, v_p), g, color='#d62728', lw=2, ls='--',
                  label='Parametric Var')
    ax_right.fill_betweenx(g, 0, norm.pdf(g, VaR_t,    v_e), alpha=0.15, color='#1f77b4')
    ax_right.fill_betweenx(g, 0, norm.pdf(g, VaR_p_tgt, v_p), alpha=0.15, color='#d62728')
    ax_right.axhline(VaR_t, color='black', lw=1.2, label='True Var')
    ax_right.set_xlabel('Density', fontsize=10)
    ax_right.legend(loc='upper right', fontsize=9)
    ax_right.grid(True, ls=':', alpha=0.6)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    title = (
        f'Joint sampling distribution of $(\\widehat{{SR}}, \\widehat{{VaR}}_{{\\alpha}})$ — {dgp_label}\n'
        f'skew = {moms["skew"]:+.2f},  excess kurt = {moms["kurt"]-3:+.2f},  '
        f'mean = {moms["mu"]:+.2f},  var = {moms["sigma"]**2:+.2f},  \n'
        f'T = {T},  n_sim = {cfg["n_sim"]} MC reps,  $\\alpha$ = {cfg["alpha"]}'
    )
    plt.suptitle(title, fontsize=12, y=0.97)
    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches='tight')
    return fig


# =============================================================================
# 4. Driver
# =============================================================================
if __name__ == '__main__':
    cfg = dict(alpha=0.01, T=500, n_sim=50000, tau=0.95, seed=7)
    dgps = {
        #'Gaussian': NormalDGP(mu=0.05, sigma=0.15),
        # 'Heavy-skew mixture': GaussianMixtureDGP(
        #     weights=[0.99, 0.01],
        #     means  =[0.082, -0.131],
        #     sigmas =[0.082,  0.245]),
        'student': StudentTDGP(mu=0.05, sigma=0.15, nu=6)
    }

    for label, dgp in dgps.items():
        print(f'\n{"="*64}\n{label}\n{"="*64}')
        res = analyze(dgp, **cfg)
        d, m = res['diagnostics'], res['moments']
        print(f"  moments  : mu={m['mu']:+.4f}  sigma={m['sigma']:.4f}  "
              f"skew={m['skew']:+.3f}  kurt={m['kurt']:.3f}")
        print(f"  truth    : SR={res['truth']['SR']:+.4f}  "
              f"VaR={res['truth']['VaR']:.4f}")
        print(f"  par.tgt  : VaR={res['asym_target']['VaR_par']:.4f}  "
              f"(asymp bias {res['asym_target']['asymp_bias_par']:+.4f})")
        print(f"  Empirical : MSE={d['mc_mse_emp']:.2e}  "
              f"bias={d['mc_bias_emp']:+.4f}  cov={d['coverage_emp']:.1%}")
        print(f"  Parametric: MSE={d['mc_mse_par']:.2e}  "
              f"bias={d['mc_bias_par']:+.4f}  cov={d['coverage_par']:.1%}")

        plot_results(res, dgp_label=label, )
        plt.show()