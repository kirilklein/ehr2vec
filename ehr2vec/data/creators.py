import itertools
import logging
import pandas as pd
from ehr2vec.data.utils import Utilities

logger = logging.getLogger(__name__)

class BaseCreator:
    def __init__(self, config: dict):
        self.config = config

    def __call__(self, concepts: pd.DataFrame, patients_info: pd.DataFrame)-> pd.DataFrame:
        return self.create(concepts, patients_info)
    
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame)-> pd.DataFrame:
        raise NotImplementedError
    
    @staticmethod
    def find_column(patients_info: pd.DataFrame, match: str):
        """Check if a column containing the match string exists in patients_info and return it."""
        columns = patients_info.columns.str.lower().str.contains(match.lower())
        if any(columns):
            return patients_info.columns[columns][0]
        raise KeyError(f'No column containing "{match}" found in patients_info')

class AgeCreator(BaseCreator):
    feature = id = 'age'
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame)-> pd.DataFrame:
        birthdate_col = self.find_column(patients_info, 'birth')
        logger.info(f'Creating age feature using birthdate column: {birthdate_col}')
        birthdates = pd.Series(patients_info[birthdate_col].values, index=patients_info['PID']).to_dict()
        # Calculate approximate age
        ages = (concepts['TIMESTAMP'] - concepts['PID'].map(birthdates)).dt.days / 365.25
        if self.config.age.get('round'):
            ages = ages.round(self.config.age.get('round'))

        concepts['AGE'] = ages
        return concepts

class AbsposCreator(BaseCreator):
    feature = id = 'abspos'
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame)-> pd.DataFrame:
        abspos = Utilities.get_abspos_from_origin_point(concepts['TIMESTAMP'], self.config.abspos)
        concepts['ABSPOS'] = abspos
        return concepts

class SegmentCreator(BaseCreator):
    feature = id = 'segment'
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame)-> pd.DataFrame:
        if 'ADMISSION_ID' in concepts.columns:
            seg_col = 'ADMISSION_ID'
        elif 'SEGMENT' in concepts.columns:
            seg_col = 'SEGMENT'
        else:
            raise KeyError('No segment column found in concepts')
    
        segments = concepts.groupby('PID')[seg_col].transform(lambda x: pd.factorize(x)[0]+1)
        
        concepts['SEGMENT'] = segments
        return concepts

class BackgroundCreator(BaseCreator):
    id = 'background'
    prepend_token = "BG_"
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame)-> pd.DataFrame:
        birthdate_column = self.find_column(patients_info, 'birth')
        # Create background concepts
        background = {
            'PID': patients_info['PID'].tolist() * len(self.config.background),
            'CONCEPT': itertools.chain.from_iterable(
                [(self.prepend_token + col + '_' +patients_info[col].astype(str)).tolist() for col in self.config.background])
        }

        if 'segment' in self.config:
            background['SEGMENT'] = 0

        if 'age' in self.config:
            background['AGE'] = -1

        if 'abspos' in self.config:
            abspos = Utilities.get_abspos_from_origin_point(patients_info[birthdate_column], self.config.abspos)
            background['ABSPOS'] = abspos.to_list() * len(self.config.background)

        # Prepend background to concepts
        background = pd.DataFrame(background)
        return pd.concat([background, concepts])

class DeathDateCreator(BaseCreator):
    id = 'death'
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame)-> pd.DataFrame:
        logger.setLevel(logging.INFO)
        birthdate_col = self.find_column(patients_info, 'birth')
        deathdate_col = self.find_column(patients_info, 'death')
        logger.info(f'Creating death feature using birthdate column: {birthdate_col} and deathdate column: {deathdate_col}')
        
        last_segments = concepts.groupby('PID')['SEGMENT'].last().to_dict()

        # Calculate age at death
        ages_at_death = (patients_info[deathdate_col] - patients_info[birthdate_col]).dt.days / 365.25
        if self.config.age.get('round'):
            ages_at_death = ages_at_death.round(self.config.age.get('round'))

        # Calculate abspos at death
        abspos_at_death = Utilities.get_abspos_from_origin_point(patients_info[deathdate_col], self.config.abspos)

        # Create death info
        death_info = {
            'PID': patients_info['PID'].tolist(),
            'CONCEPT': ['Death'] * len(patients_info),
            'AGE': ages_at_death.tolist(),
            'ABSPOS': abspos_at_death.tolist(),
            'SEGMENT': [last_segments[pid] for pid in patients_info['PID']]
        }

        # Append death info to concepts
        death_info = pd.DataFrame(death_info)
        return pd.concat([concepts, death_info])

