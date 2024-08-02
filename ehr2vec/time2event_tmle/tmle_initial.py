"""Here we follow the steps from the supplementary material of ONe step TMLE for time2event outcomes."""
import pandas as pd
import numpy as np

t_max = cls_data['t'].max()
#! todo: this needs to be changed to the survival model, currently the probability of failure/censoring is used!!

def compute_clever_covariate(patient, cls_patient, t, k, censoring_model, failure_model, A):
    """
    Compute clever covariate, see 
    Application of Time-to-Event Methods in the Assessment of Safety in Clinical Trials, Moore et al. 2009
    page 5+6

    Args:
        patient (pd.Series): The patient data.
        cls_patient (pd.DataFrame): The temporal patient data
        t (int): The time point to estimate the survival probability.
        k (int): Second time index running to T_observed.
        censoring_model (object): The fitted censoring model.
        failure_model (object): The fitted failure model.
        A (int): Which curve to estimate, 0 or 1.
    """
    if patient['A'] != A:
        return 0
    
    if k>t:
        return 0

    cls_patient_k = cls_patient[cls_patient['t'] <= k]
    cls_patient_t = cls_patient[cls_patient['t'] <= t]
    cls_patient_k_ = cls_patient[cls_patient['t'] < k]
    
    if patient['A'] == 1:
        g = patient['ps']
    else:
        g = 1 - patient['ps']

    censoring_probas = censoring_model.predict_proba(cls_patient_k_[['t', 'A', 'X']])[:, 1]
    C_surv_prob_k_ = np.prod(1 - censoring_probas)    
    
    failure_probas_t = failure_model.predict_proba(cls_patient_t[['t', 'A', 'X']])[:, 1]
    E_surv_t = np.prod(1 - failure_probas_t)

    failure_probas_k = failure_model.predict_proba(cls_patient_k[['t', 'A', 'X']])[:, 1]
    E_surv_k = np.prod(1 - failure_probas_k)

    clever_covariate = - E_surv_t/ (E_surv_k * C_surv_prob_k_ * g) 
    return clever_covariate

def compute_lambda_E(patient, failure_model, A):
    lambda_input = np.ndarray([k, A, patient['X']])
    lambda_E = failure_model.predict_proba(lambda_input)[:, 1]
    return lambda_E

def check_if_event_happened_for_positive_patient(patient, k):
    if ((patient['T_observed'] == k) and (patient['Y']==1)):
        return 1
    return 0

# Estimate Curve TMLE for the treated A = 1
TREATMENT = 1

for t in range(1, 5): # set to t_max for the full range
    j = 0 # iterations
    epsilon = 1
    while (epsilon>1e-3) and (j<3):
        h_arr = []
        N_arr = []
        lambda_E_arr = []
        for i, patient in data.iterrows():#
            cls_patient = cls_data[cls_data.pid == i]
            if i>10: # for testing purposes
                break
            for k in range(1, patient['T_observed']+1):
                N = check_if_event_happened_for_positive_patient(patient, k)
                # ! TODO: we can add ps to cls_data or simply pass the ps model for consistency.
                ht = compute_clever_covariate(patient, cls_patient, t, k, censoring_model, failure_model, A=TREATMENT) 
                
                lambda_E = compute_lambda_E(patient, failure_model, A=TREATMENT)


                
        j+=1 # to avoid infinite loops

    
#for t in time_points:
 #   print(f"t={t}")
  #  S_0 = estimate_survival_probability(cls_data, censoring_model)