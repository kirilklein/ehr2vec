import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegressionCV
from statsmodels.api import add_constant
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_linear_model import GLM


def estimate_ps(data: pd.DataFrame, covariates: list,  model: object=LogisticRegressionCV(cv=5)):
    treatment_model = model.fit(data[covariates], data['A'])
    data['propensity'] = treatment_model.predict_proba(data[covariates])[:, 1]
    return data
def compute_ipw(A, Y, ps):
    Y1_weighted = A * Y / ps
    Y0_weighted = (1 - A) * Y / (1 - ps)
    return Y1_weighted - Y0_weighted

def sample_std(estimates):
    """Compute the sample standard error of the estimate."""
    I = estimates - estimates.mean() 
    return np.sqrt((I**2).sum() / len(estimates)**2)

def IPTW_estimator(data):
    """Estimate the average treatment effect using the inverse probability of treatment weighting (IPTW) method."""
    # Estimate propensity scores
    A = data['A']
    Y = data['Y']
    data = estimate_ps(data, ['X1', 'X2'])
    # Compute weighted outcomes
    iptw = compute_ipw(A, Y, data['propensity'])
    sample_std_iptw = sample_std(iptw)
    # Estimate ATE using IPTW
    return iptw.mean(), sample_std_iptw

def estimate_outcome(data: pd.DataFrame, covariates: list, model: object=LogisticRegressionCV(cv=5)):
    X = data[covariates+['A']]
    outcome_model = model.fit(X, data['Y'])
    data['outcome'] = outcome_model.predict_proba(X)[:, 1]
    return data, model

def AIPW_estimator(data):
    """Augmented Inverse Probability of Treatment Weighting (AIPW) estimator as described in
        Glynn, Adam N., and Kevin M. Quinn. 
        "An introduction to the augmented inverse propensity weighted estimator." 
        Political analysis 18.1 (2010): 36-56.
    """
    A = data['A']
    Y = data['Y']
    data = estimate_ps(data, ['X1', 'X2'])
    data, outcome_model = estimate_outcome(data, ['X1', 'X2'])
    data = estimate_counterfactual_outcome(data, ['X1', 'X2'], outcome_model, 1)
    data = estimate_counterfactual_outcome(data, ['X1', 'X2'], outcome_model, 0)
    g = data['propensity']
    Q1 = data['outcome_1']
    Q0 = data['outcome_0']
    IPW = compute_ipw(A, Y, g)
    AIPW = IPW - (A-g)/(g*(1-g)) * ( (1-g)*Q1 + g*Q0)
    ATE_std = sample_std(AIPW)    
    return AIPW.mean(), ATE_std

def estimate_counterfactual_outcome(data: pd.DataFrame, covariates: list, fitted_model: object, treatment: int):
    """Estimate the counterfactual outcome e.g. all patients are treated or all patients are untreated."""
    data['A_temp'] = treatment
    X = data[covariates+['A_temp']]
    X = X.rename(columns={'A_temp': 'A'})
    data[f'outcome_{treatment}'] = fitted_model.predict_proba(X)[:, 1]
    del data['A_temp']
    return data

def estimate_fluctuation_parameter(data: pd.DataFrame)->float:
    """"Estimate the fluctuation parameter epsilon using a logistic regression model."""
    ps = data['propensity']
    A = data['A']
    H = A / ps - (1 - A) / (1 - ps)  
    
    # Use logit of the current outcome as offset
    offset = logit(data['outcome'])
    
    # Fit the model with offset
    model = GLM(data['Y'], add_constant(H), family=Binomial(), offset=offset).fit()
    return model.params[0], H

def tmle_initial_estimates(data):
    data = estimate_ps(data, ['X1', 'X2'])
    data, outcome_model = estimate_outcome(data, ['X1', 'X2'])
    data = estimate_counterfactual_outcome(data, ['X1', 'X2'], outcome_model, 1)
    data = estimate_counterfactual_outcome(data, ['X1', 'X2'], outcome_model, 0)
    return data

def update_Q_star(data, epsilon):
    H_1 = 1 / data['propensity']
    Q_star_1 = expit(logit(data['outcome_1']) + epsilon * H_1)
    
    H_0 = 1 / (1 - data['propensity'])
    Q_star_0 = expit(logit(data['outcome_0']) - epsilon * H_0)
    
    return Q_star_1, Q_star_0

def compute_standard_error(data, H, Q_star_1, Q_star_0, ate_tmle):
    """Compute the standard error of the average treatment effect estimate."""
    y_diff = data['Y'] - data['updated_outcome']
    IF = y_diff * H + Q_star_1 - Q_star_0 - ate_tmle
    var_IF = (IF ** 2).mean()
    return np.sqrt(var_IF/len(data))


def TMLE_estimator(data):
    """Estimate the average treatment effect using the targeted maximum likelihood estimation (TMLE) method."""
    data = tmle_initial_estimates(data)
    epsilon, H = estimate_fluctuation_parameter(data)
    Q_star_1, Q_star_0 = update_Q_star(data, epsilon)
    ate_tmle = (Q_star_1 - Q_star_0).mean()
    data['updated_outcome'] = data['A'] * Q_star_1 + (1 - data['A']) * Q_star_0
    st_error = compute_standard_error(data, H, Q_star_1, Q_star_0, ate_tmle)
    return ate_tmle, st_error


def iterative_TMLE_estimator(data, tol=1e-5, max_iter=40):
    """
    Estimate the average treatment effect using the targeted maximum likelihood estimation (TMLE) method.
    ! It does not improve over the one-shot TMLE estimator.
    """
    data = tmle_initial_estimates(data)
    
    epsilon = 0
    converged = False
    iteration = 0

    while (not converged) and (iteration < max_iter):
        print('|', end='')
        epsilon_prev = epsilon
        epsilon, H = estimate_fluctuation_parameter(data)
        Q_star_1, Q_star_0 = update_Q_star(data, epsilon)
        
        # Update the outcome model
        data['outcome_1'] = Q_star_1
        data['outcome_0'] = Q_star_0
        data['outcome'] = data['A'] * Q_star_1 + (1 - data['A']) * Q_star_0

        # Check for convergence
        if abs(epsilon - epsilon_prev) < tol:
            # print("Converged after", iteration, "iterations.")
            converged = True
        
        iteration += 1
    print('eps', epsilon)
        
    return (Q_star_1 - Q_star_0).mean()

