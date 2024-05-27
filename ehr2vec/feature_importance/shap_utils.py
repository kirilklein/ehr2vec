import numpy as np

def insert_shap_values(
        all_shap_values: np.ndarray, 
        concepts: np.ndarray, 
        shap_values: np.ndarray)->np.ndarray:
    """Insert shap values into all_shap_values"""
    ind = concepts.flatten()
    all_shap_values[ind] = (all_shap_values[ind] + shap_values.flatten())/2 # running average
    return all_shap_values