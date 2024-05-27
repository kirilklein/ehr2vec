import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def log_most_important_features(feature_importance: np.ndarray, 
                                vocabulary: Dict[str, int], 
                                num_features:int=20)->None:
    """Log the most important features based on the feature importance."""
    inv_vocab = {v: k for k, v in vocabulary.items()}
    feature_importance_dic = {inv_vocab[i]: importance for i, importance in enumerate(feature_importance)}
    sorted_features = sorted(feature_importance_dic.items(), key=lambda x: x[1], reverse=True)
    sorted_features = sorted_features[:num_features]
    logger.info("Features with largest importance")
    for feature, importance in sorted_features:
        logger.info(f"{feature}: {importance}")
    
