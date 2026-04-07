import math
import scipy.stats as stats

def wilson_interval(p_h, n, alpha):
    if n == 0:
        return (0.0, 0.0)
    if not (0.0 <= p_h <= 1.0):
        raise ValueError("p_h should be between 0 and 1.")

    z = stats.norm.ppf(1 - alpha / 2)
    z_squared = z ** 2

    denominator = 1 + (z_squared / n)

    center_adjusted = (p_h + (z_squared / (2 * n))) / denominator

    variance_term = (p_h * (1 - p_h)) / n
    adjustment_term = z_squared / (4 * (n ** 2))
    
    margin_of_error = (z / denominator) * math.sqrt(variance_term + adjustment_term)

    lower_bound = center_adjusted - margin_of_error
    upper_bound = center_adjusted + margin_of_error

    return max(0.0, lower_bound), min(1.0, upper_bound)

def acceptance_region_binomial_prop(target_p, n, alpha):#2 sided
    z = stats.norm.ppf(1 - alpha / 2)
    true_standard_error = math.sqrt((target_p * (1 - target_p)) / n)
    margin_of_error = z * true_standard_error
    
    lower_bound = target_p - margin_of_error
    upper_bound = target_p + margin_of_error
    
    return lower_bound, upper_bound
