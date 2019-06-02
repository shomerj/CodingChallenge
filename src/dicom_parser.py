import pydicom
from pydicom.errors import InvalidDicomError
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from parsing import *
import os, glob
import argparse


DEBUG = False #set true to reveal print statments

class DicomParser():
    '''
    This class will take DICOM and contour files and return or save to disk a
    pandas dataframe linking the DICOM files with their associated contour file

    Args:
        dicom_root_dir (str): Path to the DICOM root directory
        contour_root_dir (str): Path to contour root directory
        linker_csv (str): Path to link.csv file
        contour_type (str): one of either i-contour or o-contour

    '''
    def __init__(self, dicom_root_dir, contour_root_dir, linker_csv, contour_type):
        self.dicom_root_dir = dicom_root_dir
        self.contour_root_dir = contour_root_dir
        self.linker_csv = linker_csv
        self.linker_df = pd.read_csv(linker_csv) #puts linkers csv into a pandas dataframe
        self.contour_type = contour_type

        assert self.contour_type in['i-contours', 'o-contours'], 'Invalid Contour Type'


    def _get_contour_index(self, file):
        '''
        Helper function to return contour file index
        '''

        return int(file.split('-')[2])


    def _get_dicom_index(self, file):
        '''
        Helper function to return DICOM file index
        '''

        _, fname = os.path.split(file)
        return int(fname.split('.')[0])


    def create_dicom_contour_df(self, save_path=None):
        '''
        This function will create a pandas DataFrame that will like all the associated DICOM files
        with their contour file.

        input: save_path (str): if passed dataframe will be save to that specific location

        Return dicom_df (pandas DataFrame): dataframe where each row is (patient_id, DICOM file path, contour file path)
        '''

        cols = ['patient', 'dicom_path', 'contour_path']
        dicom_df = pd.DataFrame(columns=cols)
        pointer = 0 #pointer for appending to the dataframe

        for row in range(len(self.linker_df)):

            dicom, contour = self.linker_df.loc[row]
            dicom_path = os.path.abspath(os.path.join(self.dicom_root_dir, dicom))#all dicom paths for the given patient
            contour_path = os.path.abspath(os.path.join(self.contour_root_dir, contour, self.contour_type)) #all contour paths for the given patient
            contours = glob.glob(os.path.join(contour_path, '*.txt'))
            dicoms = glob.glob(os.path.join(dicom_path, '*.dcm'))
            contour_indecies = [self._get_contour_index(os.path.split(file)[-1]) for file in contours]

            DEBUG and print(f"****"*10)
            DEBUG and print(f"Patient ID {dicom}")
            DEBUG and print(f"****"*10)

            for file in dicoms:
                file = os.path.split(file)[-1]
                idx = self._get_dicom_index(file)
                if idx in contour_indecies:
                    contour_idx = contour_indecies.index(idx)

                    DEBUG and print(f'DICOM index {idx} | Contour Index {contour_indecies[contour_idx]}')

                    dicom_df.loc[pointer] = dicom, os.path.join(dicom_path, file), contours[contour_idx]


                pointer += 1

        if save_path:
            dicom_df.to_csv(os.path.join(save_path, f'{self.contour_type}_data.csv'), index=False)

        return dicom_df


    def return_dicom_img_bool_mask(self, dicom_file, contour_file):
        '''
        Given a DICOM file and contour file the will return the associated
        image and boolean mask
        '''

        coords_lst = parse_contour_file(contour_file)
        dcm_dict = parse_dicom_file(dicom_file)
        image = dcm_dict['pixel_data']
        height, width = image.shape[:2]
        mask = poly_to_mask(coords_lst, width, height)

        return image, mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--dicom_data', required=True, type=str, help='Path to DICOM data root directory ')
    parser.add_argument('-cd', '--cont_data', required=True, type=str, help='Path to Contour data root directory ')
    parser.add_argument('-l', '--linker', required=True, type=str, help='Path to link.csv file')
    parser.add_argument('-s', '--save_dir', default=None, help='Enter save path if you wish to save the DICOM csv file to disk')
    parser.add_argument('--debug', action='store_true', help='Enter debug flag to have print statements shown')
    parser.add_argument('-o', '--o_contour', action='store_true', help='Enter if dealing with o-contour files')
    parser.add_argument('-i', '--i_contour', action='store_true', help='Enter if dealing with i-contour files')



    args = parser.parse_args()

    DEBUG = args.debug

    if args.o_contour:
        dp = DicomParser(args.dicom_data, args.cont_data, args.linker, 'o-contours' )
    elif args.i_contour:
        dp = DicomParser(args.dicom_data, args.cont_data, args.linker, 'i-contours' )

    dp.create_dicom_contour_df(save_path=args.save_dir)
