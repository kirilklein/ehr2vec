import logging
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ehr2vec.common.utils import Data
from ehr2vec.data.utils import Utilities

logger = logging.getLogger(__name__)


class OutcomeMaker:
    def __init__(self, config: dict, features_cfg: dict):
        self.outcomes = config.outcomes
        self.features_cfg = features_cfg
        self.config = config

    def __call__(
        self, concepts_plus: pd.DataFrame, patients_info: pd.DataFrame, patient_set: List[str]
    )->dict:
        """Create outcomes from concepts_plus and patients_info"""
        concepts_plus = self.filter_table_by_pids(concepts_plus, patient_set)
        patients_info = self.filter_table_by_pids(patients_info, patient_set)
        concepts_plus = self.remove_missing_timestamps(concepts_plus)
 
        outcome_tables = {}
        for outcome, attrs in self.outcomes.items():
            types = attrs["type"]
            matches = attrs["match"]
            if types == "patients_info":
                timestamps = self.match_patient_info(patients_info, matches)
            else:
                timestamps = self.match_concepts(concepts_plus, types, matches, attrs)
            timestamps['TIMESTAMP'] = Utilities.get_abspos_from_origin_point(timestamps['TIMESTAMP'], self.features_cfg.features.abspos) 
            timestamps['TIMESTAMP'] = timestamps['TIMESTAMP'].astype(int)
            outcome_tables[outcome] = timestamps
        return outcome_tables
    
    @staticmethod
    def filter_table_by_pids(table: pd.DataFrame, pids: List[str])->pd.DataFrame:
        return table[table.PID.isin(pids)]

    @staticmethod
    def remove_missing_timestamps(concepts_plus: pd.DataFrame )->pd.DataFrame:
        return concepts_plus[concepts_plus.TIMESTAMP.notna()]

    def match_patient_info(self, patients_info: dict, match: List[List])->pd.Series:
        """Get timestamps of interest from patients_info"""
        return patients_info[['PID', match]].dropna()

    def match_concepts(self, concepts_plus: pd.DataFrame, types: List[List], 
                       matches:List[List], attrs:Dict)->pd.DataFrame:
        """It first goes through all the types and returns true for a row if the entry starts with any of the matches.
        We then ensure all the types are true for a row by using bitwise_and.reduce. E.g. CONCEPT==COVID_TEST AND VALUE==POSITIVE"""
        if 'exclude' in attrs:
            concepts_plus = concepts_plus[~concepts_plus['CONCEPT'].isin(attrs['exclude'])]
        col_booleans = self.get_col_booleans(concepts_plus, types, matches, 
                                             attrs.get("match_how", 'startswith'), attrs.get("case_sensitive", True))
        mask = np.bitwise_and.reduce(col_booleans)
        if "negation" in attrs:
            mask = ~mask
        return concepts_plus[mask].drop(columns=['ADMISSION_ID', 'CONCEPT'])
    
    @staticmethod
    def get_col_booleans(concepts_plus:pd.DataFrame, types:List, matches:List[List], 
                         match_how:str='startswith', case_sensitive:bool=True)->list:
        col_booleans = []
        for typ, lst in zip(types, matches):
            if match_how=='startswith':
                col_bool = OutcomeMaker.startswith_match(concepts_plus, typ, lst, case_sensitive)
            elif match_how == 'contains':
                col_bool = OutcomeMaker.contains_match(concepts_plus, typ, lst, case_sensitive)
            else:
                raise ValueError(f"match_how must be startswith or contains, not {match_how}")
            col_booleans.append(col_bool)
        return col_booleans
    
    @staticmethod
    def startswith_match(df: pd.DataFrame, column: str, patterns: List[str], case_sensitive: bool) -> pd.Series:
        """Match strings using startswith"""
        if not case_sensitive:
            patterns = [x.lower() for x in patterns]
            return df[column].astype(str).str.lower().str.startswith(tuple(patterns), False)
        return df[column].astype(str).str.startswith(tuple(patterns), False)
    
    @staticmethod
    def contains_match(df: pd.DataFrame, column: str, patterns: List[str], case_sensitive: bool) -> pd.Series:
        """Match strings using contains"""
        col_bool = pd.Series([False] * len(df), index=df.index)
        for pattern in patterns:
            if not case_sensitive:
                pattern = pattern.lower()
            if case_sensitive:
                col_bool |= df[column].astype(str).str.contains(pattern, na=False) 
            else: 
                col_bool |= df[column].astype(str).str.lower().str.contains(pattern, na=False)
        return col_bool
# !TODO: finish OutcomeHandler
class OutcomeHandler:
    ORIGIN_POINT = {'year': 2020, 'month': 1, 'day': 26, 'hour': 0, 'minute': 0, 'second': 0}
    def __init__(self, 
                index_date: Dict[str, int]=None,
                select_patient_group: str=None,
                drop_pids_w_outcome_pre_followup: bool=False,
                n_hours_start_followup: int=0,
                 ):
        """
        index_date (optional): use same censor date for all patients
        select_patient_group (optional): select only exposed or unexposed patients
        drop_pids_w_outcome_pre_followup (optional): remove patients with outcome before follow-up start
        n_hours_start_followup (optional): number of hours to start follow-up after exposure (looking for positive label)
        """
        self.index_date = index_date
        self.select_patient_group = select_patient_group
        self.drop_pids_w_outcome_pre_followup = drop_pids_w_outcome_pre_followup
        self.n_hours_start_followup = n_hours_start_followup
    
    def handle(
            self,
            data: Data,
            outcomes: pd.DataFrame, 
            exposures: pd.DataFrame,
            ) -> Tuple[Dict[str, List], Dict[str, List]]:
        """
        data: Patient Data
        outcomes: DataFrame with outcome timestamps
        exposures: DataFrame with exposure timestamps

        The following steps are taken:
         1. Filter outcomes and censor outcomes to only include patients in the data
         2. Pick earliest exposure timestamp as index_date for the exposed patients
         3. Assign index timestamp to patients without it (save which patients were actually with an exposure)
         4. Optionally select only exposed/unexposed patients
         5. Optinally remove patients with outcome(s) before start of follow-up period
         6. Select first outcome after start of follow-up for each patient
         7. Assign outcome- and index dates to data.
        """
        self.check_input(outcomes, exposures)
        # Step 1: Filter to include only relevant patients
        outcomes = self.filter_outcomes_by_pids(outcomes, data, 'outcomes')
        exposures = self.filter_outcomes_by_pids(exposures, data, 'censoring timestamps')
        
        # Step 2: Pick earliest exposure ts as index date 
        index_dates = self.get_first_event_by_pid(exposures)

        # Step 3 (Optional): Use a specific index date for all
        if self.index_date:
            index_dates = self.compute_abspos_for_index_date(data.pids)

        # Step 4: Assign censoring to patients without it (random assignment)
        exposed_patients = set(index_dates.index)
        logger.info(f"Number of exposed patients: {len(exposed_patients)}")
        index_dates = self.draw_index_dates_for_unexposed(index_dates, data.pids)

        # Step 5 (Optional): Select only exposed/unexposed patients
        if self.select_patient_group:
            data = self.select_exposed_or_unexposed_patients(data, exposed_patients)
            
        # Step 6: Select first outcome after censoring for each patient
        outcomes, outcome_pre_followup_pids = self.get_first_outcome_in_follow_up(outcomes, index_dates)
        # Step 7 (Optional): Remove patients with outcome(s) before censoring
        if self.drop_pids_w_outcome_pre_followup:
            logger.info(f"Remove {len(outcome_pre_followup_pids)} patients with outcome before start of follow-up.")
            data = data.exclude_pids(outcome_pre_followup_pids)
        # Step 8: Assign outcomes and censor outcomes to data
        data = self.assign_exposures_and_outcomes_to_data(data, index_dates, outcomes)
        return data
    
    def check_input(outcomes, exposures):
        """Check that outcomes and exposures have columns PID and TIMESTAMP."""
        if 'PID' not in outcomes.columns or 'TIMESTAMP' not in outcomes.columns:
            raise ValueError("Outcomes must have columns PID and TIMESTAMP.")
        if 'PID' not in exposures.columns or 'TIMESTAMP' not in exposures.columns:
            raise ValueError("Exposures must have columns PID and TIMESTAMP.")
        
    @staticmethod
    def filter_outcomes_by_pids(outcomes: pd.DataFrame, data: Data, type_info:str='')->pd.DataFrame:
        """Filter outcomes to include only patients in the data."""
        logger.info(f"Filtering {type_info} to include only patients in the data.")
        logger.info(f"Original number of patients in outcomes: {len(outcomes.PID.unique())}")
        filtered_outcomes = outcomes[outcomes['PID'].isin(data.pids)]
        logger.info(f"Number of patients in outcomes after filtering: {len(outcomes.PID.unique())}")
        return filtered_outcomes

    @staticmethod
    def assign_exposures_and_outcomes_to_data(data: Data, exposures: pd.Series, outcomes: pd.Series)->Data:
        """Assign exposures and outcomes to data."""
        logger.info("Assigning exposures and outcomes to data.")
        data.add_outcomes(outcomes)
        data.add_index_dates(exposures)
        return data

    @staticmethod
    def select_exposed_or_unexposed_patients(data: Data, exposed_patients: set, select_patient_group: str)->Data:
        """Select only exposed or unexposed patients."""
        logger.info(f"Selecting only {select_patient_group} patients.")
        if select_patient_group == 'exposed':
            data = data.select_data_subset_by_pids(exposed_patients)
        elif select_patient_group == 'unexposed':
            data = data.exclude_pids(exposed_patients)
        else:
            raise ValueError(f"select_patient_group must be one of None, exposed or unexposed, not {select_patient_group}")
        return data

    @staticmethod
    def draw_index_dates_for_unexposed(censoring_timestamps: pd.Series, data_pids: List[str])->pd.Series:
        """Draw censor dates for patients that are not in the censor_timestamps."""
        np.random.seed(42)
        missing_pids = set(data_pids) - set(censoring_timestamps.index)
        random_abspos = np.random.choice(censoring_timestamps.values, size=len(missing_pids))
        new_entries = pd.Series(random_abspos, index=missing_pids)
        censoring_timestamps = censoring_timestamps.append(new_entries)
        return censoring_timestamps

    def compute_abspos_for_index_date(self, pids: List)->pd.Series:
        """
        Create a pandas series hlding the same abspos based on self.index_date if not None.
        """
        logger.info(f"Using {self.index_date} as index_date for all patients.")
        index_datetime = datetime(**self.index_date)
        logger.warning(f"Using {self.ORIGIN_POINT} as origin point. Make sure is the same as used for feature creation.")
        outcome_abspos = Utilities.get_abspos_from_origin_point([index_datetime], self.ORIGIN_POINT)
        return pd.Series(outcome_abspos*len(pids), index=pids)
        
    @staticmethod
    def get_first_event_by_pid(table:pd.DataFrame):
        """Get the first event for each PID in the table."""
        logger.info(f"Selecting earliest censoring timestamp for each patient.")
        return table.groupby('PID').TIMESTAMP.min()
    
    def remove_outcomes_before_start_of_follow_up(self, outcomes: pd.DataFrame, index_dates: pd.Series
                                    ) -> Tuple[pd.DataFrame, set]:
        """
        Filter the outcomes to include only those occurring at or after the censor timestamp for each PID.
        Returns: filtered dataframe, pids removed in this process.
        """
        initial_pids = set(outcomes['PID'].unique())
        # Merge outcomes with censor timestamps
        index_date_df = index_dates.rename('index_date').reset_index()
        # Merge outcomes with censor timestamps
        joint_df = outcomes.merge(index_date_df, left_on='PID', right_on='index').drop(columns=['index'])
        # Filter outcomes to get only those at or after the censor timestamp
        filtered_df = joint_df[joint_df['TIMESTAMP'] >= joint_df['index_date']+self.n_hours_start_followup]
        # Get the PIDs that were removed
        filtered_pids = set(filtered_df['PID'].unique())
        pids_w_outcome_pre_followup = initial_pids - filtered_pids

        return outcomes, pids_w_outcome_pre_followup

    def get_first_outcome_in_follow_up(self, outcomes: pd.DataFrame, index_dates: pd.Series) -> pd.Series:
        """Get the first outcome event occurring at or after the censor timestamp for each PID."""
        # First filter the outcomes based on the censor timestamps
        filtered_outcomes, outcome_pre_followup_pids = self.remove_outcomes_before_start_of_follow_up(outcomes, index_dates)
        first_outcome = self.get_first_event_by_pid(filtered_outcomes)
        return first_outcome, outcome_pre_followup_pids
    