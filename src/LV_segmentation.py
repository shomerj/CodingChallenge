import cv2
import pandas as pd
from parsing import *
from visualize_data import convert_contours
from training_pipline import return_dicom_img_bool_mask
import matplotlib.pyplot as plt



def lv_segmentation(i_con_csv, o_con_csv, threshold_vals, with_contours=True, save_fig=True):
    '''
    This function will be used as a tool for visualizing different thresholding in
    an attempt to segment the i-contours given the o-contours

    Args:
        i_con_csv (str): path to the i-contour csv file created by the dicom parser
        o_con_csv (str): path to the o-contour csv file created by the dicom parser
        threshold_vals (list): list of up to 5 different threshold values
        with_contours (bool): set True if you want to see the images show with original contours
        save_fig (bool): set True if you with to save figure

    Return:
        dicom_img (np.array): original dicom image
        threshold_images (list): list of all the threshold masks
    '''

    o_con_df = pd.read_csv(o_con_csv)
    i_con_df = pd.read_csv(i_con_csv)
    merged_df = pd.merge(i_con_df, o_con_df, how='inner', on=['dicom_path'], suffixes=['_i', '_o'])

    row = merged_df.sample(1)
    dicom_path = row.dicom_path.item()
    o_contour_path = row.contour_path_o.item()
    i_contour_path = row.contour_path_i.item()
    idx = row.index.item()

    o_contours = parse_contour_file(o_contour_path)
    i_contours = parse_contour_file(i_contour_path)
    o_contours = convert_contours(o_contours)
    i_contours = convert_contours(i_contours)


    dicom_img, mask = return_dicom_img_bool_mask(dicom_path, o_contour_path)

    if with_contours:
        dicom_img= cv2.drawContours(dicom_img.astype(np.uint8), [i_contours], -1, (0,255,0), 1)
        dicom_img = cv2.drawContours(dicom_img.astype(np.uint8), [o_contours], -1, (0,0,255), 1)

    o_contour_img = dicom_img*mask
    plt.subplot(2,3,1)
    plt.imshow(dicom_img, cmap='gray')
    plt.title('Original')
    plt.xticks([])
    plt.yticks([])

    threshold_images = []

    for i, thresh in enumerate(threshold_vals):

        _, thresh_img = cv2.threshold(o_contour_img, thresh, 255, cv2.THRESH_BINARY)
        threshold_images.append(thresh_img)

        if with_contours:
            dicom_img = cv2.drawContours(thresh_img, [o_contours], -1, (255,255,255), 1)

        plt.subplot(2,3,i+2)
        plt.imshow(thresh_img, cmap='gray')
        plt.title(f'Threshold {thresh}')
        plt.xticks([])
        plt.yticks([])

    plt.suptitle('Segmentation at Different Thresholds')

    if save_fig:
        plt.savefig(f'../images/segmentation/segmentation_{idx}.png')

    return dicom_img, threshold_images


def compare_pixel_intensities(i_contour_csv, o_contour_csv, save_img=True):
    '''
    This function is used to evaluate the pixel intensity of the heart muscle
    vs inside the blood pool. Returns plot of the image intensities

    Args:
        i_con_csv (str): path to the i-contour csv file created by the dicom parser
        o_con_csv (str): path to the o-contour csv file created by the dicom parser
        save_img (bool): set True if you with to save figure


    '''

    o_con_df = pd.read_csv(o_contour_csv)
    i_con_df = pd.read_csv(i_contour_csv)

    merged_df = merged_df = pd.merge(i_con_df, o_con_df, how='inner', on=['dicom_path'], suffixes=['_i', '_o'])
    row = merged_df.sample(1)

    #grabbing all needed paths
    i_con_path = row.contour_path_i.item()
    o_con_path = row.contour_path_o.item()
    dicom_path = row.dicom_path.item()
    idx = row.index.item()

    #grab dicom image and associated masks
    dicom_img, i_con_mask = return_dicom_img_bool_mask(dicom_path, i_con_path)
    _, o_con_mask = return_dicom_img_bool_mask(dicom_path, o_con_path)

    #prep muscle mask
    muscle_mask = o_con_mask * np.invert(i_con_mask)

    muscle_mask = muscle_mask.astype(np.uint8)
    i_con_mask = i_con_mask.astype(np.uint8)
    dicom_img = dicom_img.astype(np.uint8)


    muscle_hist = cv2.calcHist([dicom_img], [0],muscle_mask, [256], [0, 256])
    i_con_hist = cv2.calcHist([dicom_img], [0], i_con_mask, [256], [0, 256])

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(muscle_hist, label='Heart Muscle')
    ax.plot(i_con_hist, label='Inner Contour')
    ax.legend()
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')

    ax1 = fig.add_subplot(212)
    ax1.imshow(dicom_img, cmap='gray')
    plt.suptitle('Pixel Intensities of the Heart Muscle and Inside Blood Pool')

    if save_img:
        plt.savefig(f'../images/pixel_intensity/pixel_intensity_{idx}.png')
    plt.show()


def contours_from_thresholds(i_con_csv, o_con_csv, threshold_vals, with_contours=False, save_fig=True):

    '''
    This function will draw contours based on the different threshold levels to try and isolate the i-contour
    region

    Args:
        i_con_csv (str): path to the i-contour csv file created by the dicom parser
        o_con_csv (str): path to the o-contour csv file created by the dicom parser
        threshold_vals (list): list of up to 5 different threshold values
        with_contours (bool): set True if you want to see the images show with original contours
        save_fig (bool): set True if you with to save figure
    '''

    dicom_img, threshold_imgs = lv_segmentation(i_con_csv, o_con_csv, threshold_vals, with_contours=False, save_fig=False)
    dicom_img = cv2.cvtColor(dicom_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    plt.subplot(2,3,1)
    plt.imshow(dicom_img, cmap='gray')
    plt.title('Original')
    plt.xticks([])
    plt.yticks([])

    for i, thresh in enumerate(threshold_imgs):
        im2, contours, hierarchy = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        img = cv2.drawContours(dicom_img, contours, -1, (0,255,0), 1)
        plt.subplot(2,3,i+2)
        plt.imshow(img, cmap='gray')
        plt.title(f'Threshold {threshold_vals[i]}')
        plt.xticks([])
        plt.yticks([])
        del img
    plt.suptitle('Drawing Contours from Image Thresholding')

    if save_fig:
        plt.savefig(f'../images/drawing_thresholds/threshold_segmentation.png')

    plt.show()
