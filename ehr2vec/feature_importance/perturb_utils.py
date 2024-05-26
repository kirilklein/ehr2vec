import torch
from os.path import join
import logging
from typing import Dict, List
from ehr2vec.feature_importance.perturb import PerturbationModel
from ehr2vec.data.batch import Batches

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
        num_features:int=20)->None:
    """Log the most important features based on the sigmas from the perturbation model."""
    sigmas = perturbation_model.noise_simulator.sigmas_embedding.weight.flatten().cpu().detach().numpy()
    feature_importance = 1/(sigmas+1e-9)
    inv_vocab = {v: k for k, v in vocabulary.items()}
    feature_importance_dic = {inv_vocab[i]: importance for i, importance in enumerate(feature_importance)}
    sorted_features = sorted(feature_importance_dic.items(), key=lambda x: x[1], reverse=True)
    sorted_features = sorted_features[:num_features]
    logger.info("Features with largest importance")
    for feature, importance in sorted_features:
        logger.info(f"{feature}: {importance}")

def compute_concept_frequency(features:Dict, vocabulary: Dict)->torch.Tensor:
    """Compute frequency of concepts in the features."""
    frequencies = torch.ones(len(vocabulary))
    concepts = torch.tensor(Batches.flatten(features['concept']))
    sorted_concepts, empiric_frequencies = torch.unique(concepts, return_counts=True)
    frequencies[sorted_concepts] += empiric_frequencies
    return frequencies