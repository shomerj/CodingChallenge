import unittest
import pandas as pd
import numpy as np
from dicom_parser import DicomParser
import os


class TestParsing(unittest.TestCase):
    def __init__(self,*args, **kwargs):
        super(TestParsing, self).__init__(*args, **kwargs)

        self.dp = DicomParser('../final_data/dicoms', '../final_data/contourfiles', '../final_data/link.csv')
        self.df = self.dp.create_dicom_contour_df()

    def test_parse_contour_file(self):
        '''
        Testing to see if there are the correct number of labeled images in our dataframe.
        '''


        contour_files = []

        for root, dir, files in os.walk('../final_data/contourfiles'):
            if root.endswith('i-contours'):
                for file in files:
                    contour_files.append(file)

        self.assertEqual(len(self.df), len(contour_files), f"Length of df should be {len(contour_files)}")

    def test_parsing_correctly(self):
        dicom_idxs = np.zeros(len(self.df))
        contour_idxs = np.zeros(len(self.df))

        for idx, (_, row) in enumerate(self.df.iterrows()):

            dicom_path = row['dicom_path']
            contour_path = row['contour_path']
            _, fname = os.path.split(contour_path)
            dicom_idx = self.dp._get_dicom_index(dicom_path)
            contour_idx = self.dp._get_contour_index(fname)

            dicom_idxs[idx] = dicom_idx
            contour_idxs[idx] = contour_idx

        array_bool = dicom_idxs == contour_idxs

        self.assertEqual(len(self.df), np.sum(array_bool), 'There are incorrect paring between the contour files and Dicom files')


if __name__ == '__main__':
    unittest.main()
