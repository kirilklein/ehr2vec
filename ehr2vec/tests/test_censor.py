import unittest
from unittest.mock import patch
from ehr2vec.data_fixes.censor import Censorer


class TestCensorer(unittest.TestCase):

    def setUp(self):
        self.censorer = Censorer(n_hours=1)
        self.censorer.background_length = 3

    def test_censor(self):
        features = {
            'concept': [['[CLS]', 'BG_GENDER_Male', '[SEP]', 'Diagnosis1', '[SEP]', 'Diagnosis2', 'Medication1', '[SEP]']],
            'abspos': [[0, 0, 0, 1, 1, 2, 3, 3]],
            'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1]]   
        }
        index_dates = [1]
        expected_result = {
            'concept': [['[CLS]', 'BG_GENDER_Male', '[SEP]', 'Diagnosis1', '[SEP]', 'Diagnosis2']],
            'abspos': [[0, 0, 0, 1, 1, 2]],
            'attention_mask': [[1, 1, 1, 1, 1, 1]]
        }
        result = self.censorer.censor(features, index_dates)
        self.assertEqual(result, expected_result)

    def test_if_tokenized(self):
        self.assertFalse(self.censorer._identify_if_tokenized(['[CLS]', 'BG_GENDER_Male', '[SEP]', 'Diagnosis1', '[SEP]', 'Diagnosis2']))
        self.assertTrue(self.censorer._identify_if_tokenized([0, 6, 1, 7, 1, 8]))

    def test_identify_background(self):
        self.censorer.vocabulary = {'[CLS]': 0, '[SEP]': 1, 'BG_GENDER_Male': 2, 'Diagnosis1': 3, 'Diagnosis2': 4}

        background_flags = self.censorer._identify_background(['[CLS]', 'BG_GENDER_Male', '[SEP]', 'Diagnosis1', '[SEP]', 'Diagnosis2'], tokenized_flag=False)
        self.assertEqual(background_flags, [True, True, True, False, False, False])

        background_flags_tokenized = self.censorer._identify_background([0, 2, 1, 3, 1, 4], tokenized_flag=True)
        self.assertEqual(background_flags_tokenized, [True, True, True, False, False, False])

    def test_generate_censor_flags(self):
        abspos = [0, 0, 0, 1, 1, 2, 3, 3]
        patient = {'abspos': abspos}
        event_timestamp = 1
        censor_flags = self.censorer._generate_censor_flags(patient, event_timestamp)

        self.assertEqual(censor_flags, [True, True, True, True, True, True, False, False])

class TestCensorerDiagnosesAtEndOfVisit(unittest.TestCase):

    def setUp(self):
        self.censorer = Censorer(n_hours=1)
        self.censorer.background_length = 3
        self.censorer.censor_diagnoses_at_end_of_visit = True
        self.censorer.diagnoses_codes = {'Diagnosis1', 'Diagnosis2'}

        self.features = {
            'concept': [['[CLS]', 'BG_GENDER_Male', '[SEP]', 'Diagnosis1', 'Diagnosis2', '[SEP]', 'Medication1', 'Medication2','[SEP]']],
            'abspos': [[0, 0, 0, 1.9, 4, 5, 5.5, 6, 7]],
            'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1]],
            'segment': [[0, 0, 0, 1, 1, 1, 2, 2, 2]]
        }
        self.index_dates = [1]
        self.expected_result = {
            'concept': [['[CLS]', 'BG_GENDER_Male', '[SEP]', 'Diagnosis1', 'Diagnosis2']],
            'abspos': [[0, 0, 0, 1.9, 4]],
            'attention_mask': [[1, 1, 1, 1, 1]],
            'segment': [[0, 0, 0, 1, 1]]
        }

        self.patient = {k:v[0] for k,v in self.features.items()}
        self.expected_flags = [True] * 5 + [False] * 4

    def test_generate_censor_flags(self):
        censor_flags = self.censorer._generate_censor_flags(self.patient, self.index_dates[0])
        self.assertEqual(censor_flags, self.expected_flags)

    def test_censor_diagnoses_at_end_of_visit(self):
        censor_flags = [True] * 4 + [False] * 5
        new_censor_flags = self.censorer._censor_diagnoses_at_end_of_visit(self.patient, censor_flags)
        self.assertEqual(new_censor_flags, self.expected_flags)

    def test_get_diagnoses_flags(self):
        concept = ['Diagnosis1', 'Diagnosis2', 'Medication1', 'Medication2']
        diagnoses_flags = self.censorer._get_diagnoses_flags(concept)
        self.assertEqual(diagnoses_flags, [True, True, False, False])

    def test_get_last_segment(self):
        censor_flags = [True] * 4 + [False] * 3
        segments = [0, 0, 0, 1, 1, 2, 2]
        expected_segment = 1

        last_segment = self.censorer._get_last_segment(censor_flags, segments)
        self.assertEqual(last_segment, expected_segment)

    def test_return_last_index_of_element(self):
        lst = [0, 0, 1, 1, 1, 2]
        element = 1
        expected_index = 4

        last_index = self.censorer._return_last_index_of_element(lst, element)
        self.assertEqual(last_index, expected_index)

    def test_censor(self):
        result = self.censorer.censor(self.features, self.index_dates)
        self.assertEqual(result, self.expected_result)


if __name__ == '__main__':
    unittest.main()