from typing import Tuple, Optional
from core.dgp import DGP


def get_target_moments(
    sr_objective: float, 
    mu: Optional[float] = None, 
    sigma: Optional[float] = None
) -> Tuple[float, float]:
    """
    Computes a valid (mu, sigma) pair that achieves a target Sharpe Ratio.
    Assumes a risk-free rate of 0.
    
    Parameters:
    -----------
    sr_objective : float
        The target theoretical Sharpe Ratio.
    mu : float, optional
        The theoretical mean.
    sigma : float, optional
        The theoretical standard deviation.
        
    Returns:
    --------
    Tuple[float, float]
        A tuple containing (mu, sigma).
    """
    if (mu is not None) and (sigma is not None):
        raise ValueError("Provide strictly either 'mu' or 'sigma', not both.")
    if (mu is None) and (sigma is None):
        raise ValueError("You must provide either 'mu' or 'sigma'.")

    # Case 1: Sigma is provided, compute Mu
    if sigma is not None:
        if sigma <= 0:
            raise ValueError("Sigma must be strictly positive.")
        
        computed_mu = sr_objective * sigma
        return computed_mu, float(sigma)

    # Case 2: Mu is provided, compute Sigma
    if mu is not None:
        if sr_objective == 0:
            if mu == 0:
                raise ValueError("If SR is 0 and mu is 0, sigma can be any positive value. Pass sigma instead.")
            else:
                raise ValueError("An SR objective of 0 is incompatible with a non-zero mu.")
                
        computed_sigma = mu / sr_objective
        
        if computed_sigma <= 0:
            raise ValueError("The provided 'mu' and 'sr_objective' imply a negative sigma, which is invalid.")
            
        return float(mu), computed_sigma
    

def calibrate_dgp(dgp: DGP, sr_objective, mu = None, sigma = None):
    mu, sigma = get_target_moments(sr_objective, mu, sigma)    
    dgp.calibrate_params(mu, sigma)