from typing import List, Union, Dict

import pandas as pd

from ehr2vec.common.logger import TqdmToLogger
from ehr2vec.common.utils import iter_patients
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class Censorer:
    def __init__(self, n_hours: int, vocabulary:dict=None, censor_diagnoses_at_end_of_visit=False) -> None:
        """Censor the features based on the event timestamp.
        n_hours if positive, censor all items that occur n_hours after event."""
        self.n_hours = n_hours
        self.vocabulary = vocabulary
        self.background_length = None
        self.censor_diagnoses_at_end_of_visit = censor_diagnoses_at_end_of_visit
        if self.censor_diagnoses_at_end_of_visit:
            self.diagnoses_codes = [v for k, v in self.vocabulary if k.startswith('D')]
        
    def __call__(self, features: dict, index_dates: list) -> tuple:
        sample_concepts = features["concept"][0]
        self.background_length = self.compute_background_length(sample_concepts)
        
        features = self.censor(features, index_dates)
        return features
    
    def compute_background_length(self, sample_concepts) -> List[bool]:
        """Precompute background flags for the vocabulary."""
        tokenized_flag = self._identify_if_tokenized(sample_concepts)
        return sum(self._identify_background(sample_concepts, tokenized_flag))
    
    def censor(self, features: dict, index_dates: list) -> dict:
        """Censor the features based on the censor outcomes."""
        if not self.background_length:
            raise ValueError("Background length is not computed. Please call compute_background_length first.")
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
            censor_flags = self._generate_censor_flags(patient, index_timestamp)
            censor_flags[:self.background_length] = [True]*self.background_length # keep background
            for key, value in patient.items():
                patient[key] = [item for index, item in enumerate(value) if censor_flags[index]]
                
        return patient
    
    def _generate_censor_flags(self, patient: Dict[str, List], index_timestamp: float) -> List[bool]:
        """Generate flags indicating which items to censor, based on index_timestamp and self.n_hours."""
        absolute_positions = patient['abspos']
        censor_flags = [position  <= (index_timestamp + self.n_hours) for position in absolute_positions]
        if self.censor_diagnoses_at_end_of_visit:
            censor_flags = self._censor_diagnoses_at_end_of_visit(patient, censor_flags)
        return censor_flags
    
    def _censor_diagnoses_at_end_of_visit(self, patient: Dict[str, list], 
                                          censor_flags: List[bool]) -> List[bool]:
        """Include diagnoses up to the end of the visit."""
        last_segment = self._get_last_segment(censor_flags, patient['segment'])
        last_index_to_include = self._return_last_index_of_element(patient['segment'], last_segment)
        diagnoses_flags = self._get_diagnoses_flags(patient['concept'])
        new_censor_flags = self._combine_flags_with_diagnoses_flags(
            censor_flags, diagnoses_flags, last_index_to_include) 
        return new_censor_flags
    
    @staticmethod
    def _combine_flags_with_diagnoses_flags(censor_flags: List[bool], 
                                            diagnoses_flags: List[bool], 
                                            last_index_to_include:int) -> List[bool]:
        return [flag or (diagnoses_flag and (i<=last_index_to_include))    \
                            for i, (flag, diagnoses_flag) \
                                in enumerate(zip(censor_flags, diagnoses_flags))]

    def _get_diagnoses_flags(self, concept):
        return [c in self.diagnoses_codes for c in concept]

    @staticmethod
    def _get_last_segment(censor_flags: List[bool], segments: List[int])->int:
        return [seg for seg, flag in zip(segments, censor_flags) if flag][-1]
    
    @staticmethod
    def _return_last_index_of_element(lst: List[bool], element: bool)->int:
        return len(lst)-1-lst[::-1].index(element)

    def _identify_background(self, concepts: List[Union[int, str]], tokenized_flag: bool) -> List[bool]:
        """
        Identify background items in the patient's concepts.
        Return a list of booleans of the same length as concepts indicating if each item is background.
        """
        if tokenized_flag:
            bg_values = set([v for k, v in self.vocabulary.items() if k.startswith('BG_')])
            flags = [concept in bg_values for concept in concepts]
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

