'''
Data Pipeline class for training on DICOM Images with associated pixel masks
'''

import numpy as np
import pandas as pd
from parsing import *



def return_dicom_img_bool_mask(dicom_file, contour_file):
    '''
    Given a DICOM file and contour file the will return the associated
    image and boolean mask
    '''

    coords_lst = parse_contour_file(contour_file)
    print(dicom_file)
    dcm_dict = parse_dicom_file(dicom_file)
    image = dcm_dict['pixel_data']
    height, width = image.shape[:2]
    mask = poly_to_mask(coords_lst, width, height)

    return image, mask


class TrainingPipeline():
    '''
    Yeilds mini-batches of given size

    Args:
        dicom_csv (str): path to csv file that was created from dicom_parser
        batch_size (int): size of mini batch
        img_size (int): input image size
        suffle (bool): set "True" to have the data shuffled

    Returns:

        inputs (np.array): array of size (batch_size, img_size, img_size)
        labels (np.array): array of size (batch_size, img_size, img_size)
    '''

    def __init__(self, dicom_csv, batch_size=8, img_size=256, shuffle=True):
        self.dicom_csv = pd.read_csv(dicom_csv)
        self.batch_size=batch_size
        self.img_size=img_size
        self.shuffle = shuffle

        assert batch_size < len(self), f'Batch size > {len(self)}'

    def __len__(self):
        return len(self.dicom_csv)

    def __iter__(self):

        if self.shuffle:
            self.dicom_csv.sample(frac=1)

        batches = (len(self.dicom_csv)//self.batch_size)+1

        for idx in range(batches):
            batch = self.dicom_csv.iloc[self.batch_size*idx:self.batch_size*idx+self.batch_size, [1,2]].values #grabs sequential rows from data frame
            inputs = np.zeros((len(batch), self.img_size, self.img_size))
            labels = np.zeros((len(batch), self.img_size, self.img_size))

            for i, (dicom_path, contour_path) in enumerate(batch):
                img, mask = return_dicom_img_bool_mask(dicom_path, contour_path)
                inputs[i] = img
                labels[i] = mask



            yield inputs, mask
