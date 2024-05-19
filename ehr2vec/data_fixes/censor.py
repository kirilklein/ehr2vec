from typing import List, Union

import pandas as pd

from ehr2vec.common.logger import TqdmToLogger
from ehr2vec.common.utils import iter_patients
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class Censorer:
    def __init__(self, n_hours: int, vocabulary:dict=None) -> None:
        """Censor the features based on the event timestamp.
        n_hours if positive, censor all items that occur n_hours after event."""
        self.n_hours = n_hours
        self.vocabulary = vocabulary

    def __call__(self, features: dict, index_dates: list) -> tuple:
        features = self.censor(features, index_dates)
        return features
    def censor(self, features: dict, index_dates: list) -> dict:
        """Censor the features based on the censor outcomes."""
        censored_features = {key: [] for key in features}
        censor_loop = tqdm(iter_patients(features), desc='Censoring', 
            file=TqdmToLogger(logger), total=len(features["concept"]))
        for i, patient in enumerate(censor_loop):
            index_timestamp = index_dates[i]
            censored_patient = self._censor_patient(patient, index_timestamp)

            # Append to censored features
            for key, value in censored_patient.items():
                censored_features[key].append(value)

        return censored_features

    def _censor_patient(self, patient: dict, index_timestamp: float) -> dict:
        """Censor the patient's features n_hours after index_timestamp (given in abspos)."""
        if not pd.isna(index_timestamp):
            # Extract the attention mask and determine the number of non-masked items
            attention_mask = patient["attention_mask"]
            num_non_masked = attention_mask.count(1)

            # Extract absolute positions and concepts for non-masked items
            absolute_positions = patient["abspos"][:num_non_masked]
            concepts = patient["concept"][:num_non_masked]

            # Determine if the concepts are tokenized and if they are background
            tokenized_flag = self._identify_if_tokenized(concepts)
            background_flags = self._identify_background(concepts, tokenized_flag)
            
            # Determine which items to censor based on the event timestamp and background flags
            censor_flags = self._generate_censor_flags(absolute_positions, background_flags, index_timestamp)
        
            for key, value in patient.items():
                patient[key] = [item for index, item in enumerate(value) if censor_flags[index]]
                
        return patient
    
    def _generate_censor_flags(self, absolute_positions: List[float], background_flags: List[bool], index_timestamp: float) -> List[bool]:
        """Generate flags indicating which items to censor, based on index_timestamp and self.n_hours."""
        return [
            position - index_timestamp - self.n_hours <= 0 or is_background
            for position, is_background in zip(absolute_positions, background_flags)
        ]

    def _identify_background(self, concepts: List[Union[int, str]], tokenized_flag: bool) -> List[bool]:
        """
        Identify background items in the patient's concepts.
        Return a list of booleans of the same length as concepts indicating if each item is background.
        """
        if tokenized_flag:
            bg_values = set([v for k, v in self.vocabulary.items() if k.startswith('BG_')])
            flags = [concept in bg_values for concept in concepts]
            first_background = flags.index(True)
        else:
            flags = [concept.startswith('BG_') for concept in concepts]

        # Dont censor [CLS] and [SEP] tokens of background
        first_background = flags.index(True)
        if concepts[0] == '[CLS]' or concepts[0] == self.vocabulary.get('[CLS]'):
            flags[0] = True
        if concepts[first_background+1] == '[SEP]' or concepts[first_background+1] == self.vocabulary.get('[SEP]'):
            flags[first_background+1] = True

        return flags

    @staticmethod
    def _identify_if_tokenized(concepts:list) -> bool:
        """Identify if the features are tokenized."""
        return concepts and isinstance(concepts[0], int)



