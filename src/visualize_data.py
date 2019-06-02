import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dicom_parser import DicomParser
import os
from parsing import *
import cv2


def visualize_data(data_dir, contour_type, num_shown=4, save_img=True):

    contour_dir = os.path.join(data_dir, 'contourfiles')
    dicom_dir = os.path.join(data_dir, 'dicoms')
    link_file = os.path.join(data_dir, 'link.csv')

    print(contour_dir)
    print(dicom_dir)
    print(link_file)

    d_parser = DicomParser(dicom_dir, contour_dir, link_file, contour_type)
    dicom_df = d_parser.create_dicom_contour_df()

    fig = plt.figure()
    count = 1
    for j in range(1, num_shown+1):
        idx = np.random.choice(len(dicom_df))
        dicom = dicom_df.iloc[idx, 1]
        contour = dicom_df.iloc[idx, 2]
        img, mask = d_parser.return_dicom_img_bool_mask(dicom, contour)
        seg = img*mask
        images = [img, mask, seg]
        print(img.shape)

        for i in range(3):

            ax = fig.add_subplot(num_shown, 3, count )
            ax.imshow(images[i], cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            count +=1

    plt.suptitle('Visualize Original Image, Mask and Segmentation')

    if save_img ==True:
        plt.savefig(f'../images/{contour_type}_vis.png', dpi=300)

    plt.show()


def draw_contours(data_dir=None, data=None, save_img=True):
    '''
    This is a visualization tool for visualizing both i-contours and o-contours
    '''

    contour_dir = os.path.join(data_dir, 'contourfiles')
    dicom_dir = os.path.join(data_dir, 'dicoms')
    link_file = os.path.join(data_dir, 'link.csv')


    i_con_parser = DicomParser(dicom_dir, contour_dir, link_file, 'i-contours')
    i_con_df = i_con_parser.create_dicom_contour_df()

    o_con_parser = DicomParser(dicom_dir, contour_dir, link_file, 'o-contours')
    o_con_df = o_con_parser.create_dicom_contour_df()

    merged_df = pd.merge(i_con_df, o_con_df, how='inner', on=['dicom_path'], suffixes=['_i', '_o'])

    #grabbing image and both contours from random sample in dataframe
    row = merged_df.sample(1)
    o_contour_path = row.contour_path_o.item()
    i_contour_path = row.contour_path_i.item()
    dicom_path = row.dicom_path.item()
    idx = row.index.item()

    o_contours = parse_contour_file(o_contour_path)
    i_contours = parse_contour_file(i_contour_path)

    o_contours = convert_contours(o_contours)
    i_contours = convert_contours(i_contours)

    dicom_img, _ = i_con_parser.return_dicom_img_bool_mask(dicom_path, i_contour_path)
    dicom_img = cv2.cvtColor(dicom_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    #drawing contours
    dicom_img= cv2.drawContours(dicom_img, [i_contours], -1, (0,255,0), 1)
    dicom_img = cv2.drawContours(dicom_img, [o_contours], -1, (0,0,255), 1)

    if save_img:
        cv2.imwrite(f'../images/contour_img{idx}.png', dicom_img)

    cv2.imshow('Drawn Contours', dicom_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def convert_contours(contours):
    '''
    Helper function to visualize contours
    '''
    contour_array = np.zeros((len(contours),2), dtype=int)

    for idx, contour in enumerate(contours):
        contour_array[idx] = contour[0], contour[1]

    return contour_array
