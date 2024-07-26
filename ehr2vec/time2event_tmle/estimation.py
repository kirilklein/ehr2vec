import pandas as pd
import numpy as np

def get_indicator(t, T_observed, Y, model_type):
    """Determine the event or censoring indicator based on model type."""
    if model_type == 'failure':
        return 1 if (t == T_observed and Y == 1) else 0
    elif model_type == 'censoring':
        return 1 if (t == T_observed and Y == 0) else 0
    else:
        raise ValueError("model_type must be either 'failure' or 'censoring'")

def transform_data_for_model_estimation(data, model_type='failure'):
    """Transform the data for either the failure model or the censoring model estimation.
    
    Args:
        data (pd.DataFrame): The input data containing T_observed, Y, A, X.
        model_type (str): Either 'failure' or 'censoring' to specify which model to transform the data for.
    
    Returns:
        pd.DataFrame: The transformed data suitable for classification.
    """
    expanded_rows = []

    # Loop over each patient in the original data
    for index, row in data.iterrows():
        T = int(row['T_observed'])
        expanded_rows.extend([
            {
                'pid': index,
                't': t,
                'A': row['A'],
                'X': row['X'],
                'Y_E': get_indicator(t, T, row['Y'], 'failure'),
                'Y_C': get_indicator(t, T, row['Y'], 'censoring')
            }
            for t in range(0, T + 1)
        ])

    # Create DataFrame from the list of rows
    expanded_data = pd.DataFrame(expanded_rows)
    return expanded_data

def estimate_survival_probability(data, model):
    """
    Estimate the survival curve for a given model.
    It can be censoring or failure event model.    
    Args:
        data (pd.DataFrame): The input data containing T_observed, Y, A, X.
        model (object): The fitted censoring model.
    
    Returns:
        pd.DataFrame: The input data with an additional column 'censoring_probability'.
    """
    # Prepare the input features for prediction
    X = data[['t', 'A', 'X']]
    
    # Predict the failure probabilities
    predicted_failure_probas = model.predict_proba(X)[:, 1]
    
    # Create a new DataFrame for predictions to avoid modifying the original data
    predictions = pd.DataFrame({'pid': data['pid'], 'predicted_failure_proba': predicted_failure_probas})
    
    # Group by 'pid' and compute the survival probability
    # survival_probs = predictions.groupby('pid')['predicted_failure_proba'].apply(lambda x: np.prod(1 - x))
    survival_probs = (1 - predictions.groupby('pid')['predicted_failure_proba'].transform('cumprod')).groupby(predictions['pid']).last()
    return survival_probs.values


def select_pos_patients_at_risk(cls_data, t_prime):
    event_mask = cls_data['Y_E'] == 1
    event_data = cls_data[event_mask]
    pos_pids_at_risk = event_data[event_data['t']>t_prime]['pid'].unique() # patients with an event at t_prime or later
    return cls_data[cls_data.pid.isin(pos_pids_at_risk)]

def select_treated(cls_data):
    return cls_data[cls_data['A']==1]

def select_events_up_to_t_prime(cls_data, t_prime):
    return cls_data[cls_data['t']<t_prime]

def compute_curve_for_treated_at_t(N_patients, ps, survival_probs):
    return 1/N_patients * np.sum(1 / (ps * survival_probs))

# from time import time
def IPCW_estimator(cls_data, data, censoring_model):
    """Here we estimate the survival curve for treated patients using IPCW estimator.
    For each time t, we select patients with outcome but still at risk at time t.
    We then compute for those the censoring survival probability at the time of outcome and the propensity score.
    """
    survival_curve = []
    max_time = cls_data['t'].max()
    censoring_survival_probs = estimate_survival_probability(
            cls_data, censoring_model)
    
    for t_prime in range(1,max_time):
        # print('t', t_prime)
        # 1. select patients with outcome but still at risk at time t
        cls_data_t_prime = select_pos_patients_at_risk(cls_data, t_prime)
        cls_data_t_prime = select_treated(cls_data_t_prime)
        # 2. Take only event up to time t_prime
        cls_data_t_prime = select_events_up_to_t_prime(cls_data_t_prime, t_prime)         
        selected_ps = data.iloc[cls_data_t_prime['pid'].unique()]['propensity']
        selected_survival_probs = censoring_survival_probs[cls_data_t_prime['pid'].unique()]
        # Filter out zeros in ps and survival_probs
        N = cls_data['pid'].nunique()
        survival_curve_at_t_prime = compute_curve_for_treated_at_t(N, selected_ps, selected_survival_probs)
        assert len(selected_ps)==len(selected_survival_probs), 'Length mismatch'
        survival_curve.append(survival_curve_at_t_prime)
    return np.array(survival_curve), max_time