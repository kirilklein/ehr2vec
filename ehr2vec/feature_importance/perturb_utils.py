import torch
from os.path import join
import logging
from typing import Dict
from ehr2vec.feature_importance.perturb import PerturbationModel

logger = logging.getLogger(__name__)


def average_sigmas(fi_folder:str, n_splits:int)->torch.Tensor:
    """
    Load and average sigmas from all folds. 
    Save sigmas_average.pt to fi_folder, return the averaged sigmas.
    """
    sigmas = []
    for fold in range(1, n_splits+1):
        sigmas_tensor = torch.load(join(fi_folder, f'sigmas_fold_{fold}.pt'))
        sigmas.append(sigmas_tensor)
    sigmas = torch.stack(sigmas).mean(dim=0)
    return sigmas

def log_most_important_features(
        perturbation_model:PerturbationModel, 
        vocabulary:Dict[str, int], 
        num_features:int=10)->None:
    """Log the most important features based on the sigmas from the perturbation model."""
    sigmas = perturbation_model.noise_simulator.sigmas_embedding.weight.flatten().cpu().detach().numpy()
    feature_importance = 1/(sigmas+1e-9)
    inv_vocab = {v: k for k, v in vocabulary.items()}
    feature_importance_dic = {inv_vocab[i]: importance for i, importance in enumerate(feature_importance)}
    sorted_features = sorted(feature_importance_dic.items(), key=lambda x: x[1], reverse=True)
    sorted_features = sorted_features[:num_features]
    logger.info("Features with largest importance")
    logger.info(sorted_features)
