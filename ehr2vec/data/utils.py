import os
import re
import numpy as np
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Union, Generator

from ehr2vec.common.config import Config
from ehr2vec.common.utils import Data

logger = logging.getLogger(__name__)  # Get the logger for this module

PID_KEY = 'PID'
BG_GENDER_KEYS = {
    'male': ['M', 'Mand',  'male', 'Male', 'man', 'MAN', '1'],
    'female': ['W', 'Kvinde', 'F', 'female', 'Female', 'woman', 'WOMAN', '0']
}
MIN_POSITIVES = {'finetune': 1, None: 1}
CHECKPOINT_FOLDER = 'checkpoints'



class Utilities:
    @classmethod
    def process_data(cls, data: Data, func: callable, log_positive_patients_num: bool = False, args_for_func: Dict={})->Dict:
        """Apply a function to all datasets in a dictionary"""
        data = func(data, **args_for_func)
        cls.log_patient_nums(func.__name__, data)
        if log_positive_patients_num:
            cls.log_pos_patients_num(data)

        return data

    @staticmethod
    def log_patient_nums(operation:str, data: Data)->None:
        logger.info(f"After applying {operation}:")
        logger.info(f"{len(data.pids)} patients")

    @staticmethod
    def get_abspos_from_origin_point(timestamps: Union[pd.Series, List[datetime]], 
                                     origin_point: Dict[str, int])->Union[pd.Series, List[float]]:
        """Get the absolute position in hours from the origin point"""
        origin_point = datetime(**origin_point)
        abspos = Utilities.get_relative_timestamps_in_hours(timestamps, origin_point)

        return abspos

    @staticmethod
    def get_relative_timestamps_in_hours(timestamps:Union[pd.Series, List[datetime]], 
                                         origin_point: datetime)->Union[pd.Series, List[float]]:
        """Get the relative position in hours from the origin point"""
        if isinstance(timestamps, pd.Series):
            return (timestamps - origin_point).dt.total_seconds() / 60 / 60
        elif isinstance(timestamps, list):
            return [(timestamp - origin_point).total_seconds() / 60 / 60 for timestamp in timestamps]

    @staticmethod
    def check_and_adjust_max_segment(data: Data, model_cfg: Config)->None:
        """Check max segment. If it's larger or equal to the model type vocab size, change accordingly."""
        max_segment = max([max(seg) for seg in data.features['segment']])
        type_vocab_size = model_cfg.type_vocab_size

        if max_segment >= type_vocab_size:
            logger.warning(f"You've set type_vocab_size too low. Max segment {max_segment} >= type_vocab_size {type_vocab_size}\
                             We'll change it to {max_segment+1}.")
            model_cfg.type_vocab_size = max_segment+1

    @staticmethod
    def get_token_to_index_map(vocabulary:dict)->Tuple[dict]:
        """
        Creates a new mapping from vocbulary values to new integers excluding special tokens
        """
        filtered_tokens = [v for k, v in vocabulary.items() if not k.startswith('[')]
        token2index = {token: i for i, token in enumerate(filtered_tokens)}        
        new_vocab = {k: token2index[v] for k, v in vocabulary.items() if v in token2index}

        return token2index, new_vocab

    @staticmethod
    def get_gender_token(vocabulary: dict, gender_key: str)->int:
        """Get the token from the vocabulary corresponding to the gender provided in the config"""
        # Determine the gender category
        gender_category = None
        for category, keys in BG_GENDER_KEYS.items():
            if gender_key in keys:
                gender_category = category
                break

        if gender_category is None:
            raise ValueError(f"Unknown gender {gender_key}, please select one of {BG_GENDER_KEYS}")

        # Check the vocabulary for a matching token
        for possible_key in BG_GENDER_KEYS[gender_category]:
            gender_token = vocabulary.get('BG_GENDER_' + possible_key, None)
            if gender_token is not None:
                return gender_token
    
        raise ValueError(f"None of BG_GENDER_+{BG_GENDER_KEYS[gender_category]} found in vocabulary.")

    @staticmethod
    def get_background_indices(data: Data)->List[int]:
        """Get the length of the background sentence"""
        background_tokens = set([v for k, v in data.vocabulary.items() if k.startswith('BG_')])
        if len(background_tokens)==0:
            logger.warning("No background tokens found in vocabulary")
            return []

        example_concepts = data.features['concept'][0] # Assume that all patients have the same background length
        background_indices = [i for i, concept in enumerate(example_concepts) if concept in background_tokens]

        if data.vocabulary['[SEP]'] in example_concepts:
            background_indices.append(max(background_indices)+1)

        return background_indices

    @staticmethod
    def code_starts_with(code: int, prefixes: tuple)->bool:
        """Check if the code starts with any of the given prefixes."""
        return code.startswith(prefixes)

    @staticmethod
    def log_pos_patients_num(data: Data)->None:
        num_positive_patiens = len([t for t in data.outcomes if not pd.isna(t)])
        if num_positive_patiens < MIN_POSITIVES[data.mode]:
            raise ValueError(f"Number of positive patients is less than {MIN_POSITIVES[data.mode]}: {num_positive_patiens}")
        logger.info(f"Positive {data.mode} patients: {num_positive_patiens}")

    @staticmethod
    def get_last_checkpoint_epoch(checkpoint_folder: str)->int:
        """Returns the epoch of the last checkpoint."""
        # Regular expression to match the pattern retry_XXX
        pattern = re.compile(r"checkpoint_epoch(\d+)_end\.pt$")
        max_epoch = None
        for filename in os.listdir(checkpoint_folder):
            match = pattern.match(filename)
            if match:
                epoch = int(match.group(1))
                if max_epoch is None or epoch > max_epoch:
                    max_epoch = epoch
        # Return the folder with the maximum retry number
        if max_epoch is None:
            raise ValueError("No checkpoint found in folder {}".format(checkpoint_folder))

        return max_epoch

    @staticmethod
    def split_train_val(features: Dict[str, List], pids: List, val_ratio: float = 0.2)->Tuple[Dict, List, Dict, List]:
        """Split features and pids into train and val sets.
        Returns:
            train_features: A dictionary of lists of features for training
            train_pids: A list of patient IDs for training
            val_features: A dictionary of lists of features for validation
            val_pids: A list of patient IDs for validation
        """
        val_size = int(len(features['concept']) * val_ratio)
        train_size = len(features['concept']) - val_size
        train_features = {k: v[:train_size] for k, v in features.items()}
        val_features = {k: v[train_size:] for k, v in features.items()}
        train_pids = pids[:train_size]
        val_pids = pids[train_size:]

        return train_features, train_pids, val_features, val_pids

    @staticmethod
    def iter_patients(data) -> Generator[dict, dict, dict]:
        """Iterate over patients in a features dict."""
        for i in range(len(data.features["concept"])):
            yield {key: values[i] for key, values in data.features.items()}, {key: values[i] for key, values in data.outcomes.items()}

    @staticmethod
    def censor(patient: dict, event_timestamp: float) -> dict:
        """Censor the patient's features based on the event timestamp."""
        if not pd.isna(event_timestamp):
            # Extract absolute positions and concepts for non-masked items
            absolute_positions = patient["abspos"]
            concepts = patient["concept"]
            # Determine which items to censor based on the event timestamp and background flags
            censor_flags = Utilities._generate_censor_flags(absolute_positions, event_timestamp)
        
            for key, value in patient.items():
                patient[key] = [item for index, item in enumerate(value) if censor_flags[index]]

        return patient

    @staticmethod
    def _generate_censor_flags(absolute_positions: List[float], event_timestamp: float) -> List[bool]:
        """Generate flags indicating which items to censor."""
        return [
            position - event_timestamp <= 0 for position in absolute_positions
        ]

    @staticmethod
    def select_and_reorder_feats_and_pids(feats: Dict[str, List], pids: List[str], select_pids: List[str])->Tuple[Dict[str, List], List[str]]:
        """Reorders pids and feats to match the order of select_pids"""
        if not set(select_pids).issubset(set(pids)):
            raise ValueError(f"There are {len(set(select_pids).difference(set(pids)))} select pids not present in pids") 
        pid2idx = {pid: index for index, pid in enumerate(pids)}
        indices_to_keep = [pid2idx[pid] for pid in select_pids] # order is important, so keep select_pids as list
        selected_feats = {}
        for key, value in feats.items():
            selected_feats[key] = [value[idx] for idx in indices_to_keep]
        return selected_feats, select_pids
    
    @staticmethod
    def get_segments_one_hot(segments):
        """
        One-hot encode the segments.
        Example: 
        segments = [1, 2, 3, 1, 2, 3]
        one_hot_matrix = [[1, 0, 0, 1, 0, 0],
                          [0, 1, 0, 0, 1, 0],
                          [0, 0, 1, 0, 0, 1]]
        """
        # Identify unique segments and create a mapping from segment to row index
        unique_segments, inverse_indices = np.unique(segments, return_inverse=True)
        # Create a zero matrix
        one_hot_matrix = np.zeros((len(unique_segments), len(segments)), dtype=int)
        # Row indices for one-hot encoding, using the mapping
        row_indices = inverse_indices
        # Column indices for one-hot encoding
        col_indices = np.arange(len(segments))
        # Use advanced indexing to set the appropriate elements to 1
        one_hot_matrix[row_indices, col_indices] = 1
        return one_hot_matrix

    @staticmethod
    def calculate_ages_at_censor_date(data: Data) -> List[int]:
        """
        Calculates the age of patients at their respective censor dates.
        """
        ages_at_censor_date = []
        
        for abspos, age, censor_date in zip(data.features['abspos'], data.features['age'], data.index_dates):
            if censor_date is None:
                ages_at_censor_date.append(age[-1]) # if no censoring, we take the last age
                continue
            # Calculate age differences and find the closest abspos index to the censor date
            time_differences_h = np.array([censor_date - ap for ap in abspos])
            # compute closest index (with regards to abspos) on the left to censor date
            closest_abspos_index = np.argmin(
                np.where(time_differences_h < 0, np.inf, time_differences_h)) 
            age_at_censor = age[closest_abspos_index] + time_differences_h[closest_abspos_index] / 24 / 365.25
            ages_at_censor_date.append(age_at_censor)
        return ages_at_censor_date
    @staticmethod
    def get_first_non_special_token_index(concepts: List[int], special_tokens: set) -> List[int]:
        """
        Get the first non-special token for each patient.
        """
        return next((i for i, token in enumerate(concepts) if token not in special_tokens), -1)
    @staticmethod
    def calculate_trajectory_lengths(data: Data) -> List[float]:
        """
        Calculate the lengths of the trajectories in days for each patient.
        """
        trajectory_lengths = []
        special_tokens = set([data.vocabulary[token] for token in data.vocabulary\
                               if token.startswith(('[', 'BG_'))])
        for abspos, concept, censor_date in zip(data.features['abspos'], data.features['age'], data.index_dates):
            first_concept_index = Utilities.get_first_non_special_token_index(concept, special_tokens)
            trajectory_length_hours = censor_date - abspos[first_concept_index]
            trajectory_length_days = trajectory_length_hours / 24 
            trajectory_lengths.append(trajectory_length_days)
        return trajectory_lengths
    
    @staticmethod
    def calculate_sequence_lengths(data: Data) -> List[int]:
        """Calculate the lengths of the sequences for each patient."""
        sequence_lengths = []
        special_tokens = set([data.vocabulary[token] for token in data.vocabulary\
                               if token.startswith(('[', 'BG_'))])
        for concept in data.features['concept']:
            non_special_tokens = [token for token in concept if token not in special_tokens]
            sequence_lengths.append(len(non_special_tokens))
        return sequence_lengths
