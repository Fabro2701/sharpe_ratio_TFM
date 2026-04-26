
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