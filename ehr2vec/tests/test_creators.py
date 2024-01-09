import unittest
import pandas as pd
from datetime import datetime
from data.creators import BaseCreator, AgeCreator, AbsposCreator, SegmentCreator, BackgroundCreator

class Config:
    def __init__(self):
        self.age = {'round': 2}
        self.abspos = {'year': 2020, 'month': 1, 'day': 26}
        self.segment = True
        self.background = ['GENDER']

    def __contains__(self, item):
        return self.__getattribute__(item)

class TestBaseCreator(unittest.TestCase):
    def setUp(self):
        self.config = Config()

        self.concepts = pd.DataFrame({
            'PID': ['1', '2', '3', '1'],
            'CONCEPT': ['DA1', 'DA2', 'MA1', 'DA2'],
            'TIMESTAMP': pd.to_datetime(['2020-01-02', '2021-03-20', '2022-05-08', '2023-01-02'])
        })
        self.patients_info = pd.DataFrame({
            'PID': ['1', '2', '3'],
            'BIRTHDATE': pd.to_datetime(['2000-01-02', '2000-03-20', '2000-05-08']),
            'GENDER': ['Male', 'Female', 'Male']
        })

    def test_birthdate_column(self):
        basecreator = BaseCreator(self.config)
        patients_info_pure = pd.DataFrame({'BIRTHDATE': []})
        patients_info_convert = pd.DataFrame({'DATE_OF_BIRTH': []})
        patients_info_false = pd.DataFrame({'NOTHING': []})
        self.assertIn('BIRTHDATE', basecreator._rename_birthdate_column(patients_info_pure))
        self.assertIn('BIRTHDATE', basecreator._rename_birthdate_column(patients_info_convert))
        with self.assertRaises(KeyError):
            basecreator._rename_birthdate_column(patients_info_false)

    def test_age_creator(self):
        creator = AgeCreator(self.config)
        result = creator.create(self.concepts, self.patients_info)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('AGE', result.columns)
        self.assertEqual(result.AGE.tolist(), [20, 21, 22, 23])

    def test_abspos_creator(self):
        creator = AbsposCreator(self.config)
        result = creator.create(self.concepts, self.patients_info)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('ABSPOS', result.columns)
        origin_point = datetime(**self.config.abspos)
        self.assertTrue(result.ABSPOS.iloc[0] < 0)
        self.assertTrue(all(result.ABSPOS.iloc[1:] > 0))
        self.assertEqual(result.ABSPOS.tolist(), ((self.concepts.TIMESTAMP - origin_point).dt.total_seconds() / 60 / 60).tolist())

    def test_segment_creator(self):
        creator = SegmentCreator(self.config)
        self.concepts['ADMISSION_ID'] = ['1', '2', '3', '4']
        result = creator.create(self.concepts, self.patients_info)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('SEGMENT', result.columns)
        self.assertEqual(result.SEGMENT.tolist(), [1, 1, 1, 2])

    def test_background_creator(self):
        creator = BackgroundCreator(self.config)
        result = creator.create(self.concepts, self.patients_info)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('SEGMENT', result.columns)
        self.assertIn('AGE', result.columns)
        self.assertIn('ABSPOS', result.columns)
        for _, patient in result.groupby("PID"):
            self.assertEqual(patient.CONCEPT.str.startswith('BG_').sum(), len(self.config.background))

if __name__ == '__main__':
    unittest.main()
