
def garch_kurtosis(k_e, alpha, beta):
    num = 1 - (alpha+beta)**2 
    den = 1 - (alpha+beta)**2 - alpha**2*(k_e-1)
    return k_e * num / den

def ar_garch_kurtosis_from_u(k_u, rho, alpha, beta):
    phi = rho
    phi2 = phi**2
    den_com = 1 - 2*alpha*beta - beta**2
    num_A = 1 - alpha*beta - beta**2
    factor_A = 1 - phi2 * (alpha + beta)
    A = 6 * phi2 * alpha * (num_A / den_com) * (1 / factor_A)

    return (1 - phi2) / (1 + phi2) * ((k_u - 1)*A + 6*phi2/(1-phi2) + k_u)

def ar_garch_kurtosis_from_e(k_e, rho, alpha, beta):
    k_u = garch_kurtosis(k_e, alpha, beta)
    return ar_garch_kurtosis_from_u(k_u, rho, alpha, beta)


#general innov ep
def ar_garch_moments(skew_eps, kurt_eps, phi, alpha, beta):
    if not (abs(phi) < 1):
        raise ValueError("AR(1) non-stationary: need |phi| < 1.")
    if alpha < 0 or beta < 0:
        raise ValueError("GARCH coefficients must be non-negative.")
    rho = alpha + beta
    if not (rho < 1):
        raise ValueError("GARCH non-stationary: need alpha + beta < 1."
                         f"alpha={alpha}, beta={beta}")
    ku_denom = 1.0 - rho**2 - alpha**2 * (kurt_eps - 1.0)
    if ku_denom <= 0:
        raise ValueError(
            "Fourth moment of u_t does not exist: "
            "1 - (alpha+beta)^2 - alpha^2 * (kurt_eps - 1) must be > 0."
            f"alpha={alpha}, beta={beta}"
        )

    # ---- skewness of returns -------------------------------------------------
    # skew_r ~ (1-phi^2)^{3/2}/(1-phi^3) * (1 + 2*phi*alpha - phi*beta) /
    #         (1 - phi*rho) * skew_eps
    one_minus_phi2 = 1.0 - phi**2
    one_minus_phi3 = 1.0 - phi**3
    tau = 1.0 - phi * rho                       # 1 - phi*(alpha+beta)
    num_skew = (one_minus_phi2 ** 1.5) * (1.0 + 2.0 * phi * alpha - phi * beta)
    den_skew = one_minus_phi3 * tau
    skew_r = num_skew / den_skew * skew_eps

    # ---- kurtosis of u_t (exact) --------------------------------------------
    k_u = kurt_eps * (1.0 - rho**2) / ku_denom

    # ---- kurtosis of returns -------------------------------------------------
    # auxiliaries
    Q = 1.0 - phi**2 * rho
    P = alpha * (1.0 - alpha * beta - beta**2) / (1.0 - 2.0 * alpha * beta - beta**2)

    term_const  = 6.0 * phi**2
    term_ku     = one_minus_phi2 * k_u
    term_garch  = 6.0 * phi**2 * one_minus_phi2 * P * (k_u - 1.0) / Q
    term_skew   = (6.0 * phi * alpha * one_minus_phi2
                   * (1.0 + phi**2 * (2.0 * alpha - beta))
                   / (Q * tau)) * skew_eps**2

    kurt_r = (term_const + term_ku + term_garch + term_skew) / (1.0 + phi**2)

    return skew_r, kurt_r