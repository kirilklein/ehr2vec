"""Simulation from R code from appendix in the paper 
    'One-step targeted maximum likelihood estimation for time-to-event outcomes' 
    !The formulas provided in the paper differ from that using a log-normal for T"""
import numpy as np
# Load the data
def simulate_covariates(n, min=0, max=1.5):
    return np.random.uniform(min, max, n)

def simulate_treatment(n, X, offset=0.4, multiplier=0.5, min_X=0.75):
    prob = offset + multiplier * (X > min_X)
    return np.random.binomial(1, prob, n)

def simulate_outcome_time(n, X, A, rate_offset=1,
                          rate_X_mult=0.7,
                           rate_A_mult=0.8):
    rate = rate_offset + rate_X_mult * X**2 - rate_A_mult * A
    Trexp = np.random.exponential(1/rate, n)
    return np.round(Trexp * 2)

def simulate_censoring(n, X, shape_offset=1,  shape_multiplier=0.5, scale=75):
    shape = shape_offset + shape_multiplier * X
    Cweib = np.random.weibull(shape, n) * scale
    return np.round(Cweib * 2)

def theoretical_survival_function(t, X, A, rate_offset=1, rate_X_mult=0.7, rate_A_mult=0.8):
    # Compute the rate parameter lambda
    rate = rate_offset + rate_X_mult * X**2 - rate_A_mult * A
    # Compute the survival probability
    survival_prob = np.exp(-rate[:, np.newaxis] * t / 2)
    # Average the survival probabilities over the population
    return np.mean(survival_prob, axis=0)

def simulate_data(n, A=None, 
                  covariate_kwargs={},
                  treatment_kwargs={}, 
                  censoring_kwargs={}, 
                  outcome_kwargs={}):
    """
    Simulate Data using model from appendix (R code) in the paper
    One-step targeted maximum likelihood estimation for time-to-event outcomes
    by Mark J. van der Laan
    """
    np.random.seed(42)
    # Covariates and Treatment
    X = simulate_covariates(n, **covariate_kwargs)
    if A is None:
        A = simulate_treatment(n, X, **treatment_kwargs)
    else:
        assert (A == 0) | (A == 1)
        A = np.ones(n) * A
    # Time to event
    T = simulate_outcome_time(n, X, A, **outcome_kwargs)
    # Censoring time
    C = simulate_censoring(n, X, **censoring_kwargs)
    # Observed time and event indicator
    T_observed = np.minimum(T, C)
    Y = (T <= C).astype(int)
    data = {'X': X, 'A': A, 'T': T, 'C': C, 'T_observed': T_observed, 'Y': Y}
    return data