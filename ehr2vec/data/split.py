import logging
from typing import Iterator, Tuple, List

from ehr2vec.common.utils import Data
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)       


def get_n_splits_cv(data: Data, n_splits: int, indices:list=None
                    )->Iterator[Tuple[List[int], List[int]]]:
    """
    Generate indices for n_splits cross-validation.

    Parameters:
    data (Data): Data object containing 'pids'.
    n_splits (int): Number of folds for cross-validation.
    indices (List[int], optional): List of indices to be used. If None, all indices from data are used.

    Yields:
    Iterator[Tuple[List[int], List[int]]]: Iterator over tuples of train and validation indices for each fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    if indices is None:
        logger.info("Using all indices")
        indices = list(range(len(data.pids)))
    # Iterate over each fold generated by KFold
    for train_idx, val_idx in kf.split(indices):
        # Map the relative indices to actual indices in the data
        train_indices = [indices[i] for i in train_idx]
        val_indices = [indices[i] for i in val_idx]

        yield train_indices, val_indices

def split_indices_into_train_val(indices:List[int], val_split:float=0.2
                                 )->Tuple[List[int], List[int]]:
    """Split indices into train and validation indices."""
    train_indices = indices[:int(len(indices)*(1-val_split))]
    val_indices = indices[int(len(indices)*(1-val_split)):]
    return train_indices, val_indices

def split_data_into_train_val(data: Data, val_split:float=0.2
                              )->Tuple[Data, Data]:
    """Split data into train and val"""
    indices = [i for i in range(len(data.pids))]
    train_indices, val_indices = split_indices_into_train_val(indices, val_split) 
    train_data = data.select_data_subset_by_indices(train_indices, mode='train')
    val_data = data.select_data_subset_by_indices(val_indices, mode='val')
    return train_data, val_data
    

