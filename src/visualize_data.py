import matplotlib.pyplot as plt
import numpy as np
from dicom_parser import DicomParser
import os


def visualize_data(data_dir, num_shown=4, save_img=True):

    contour_dir = os.path.join(data_dir, 'contourfiles')
    dicom_dir = os.path.join(data_dir, 'dicoms')
    link_file = os.path.join(data_dir, 'link.csv')

    print(contour_dir)
    print(dicom_dir)
    print(link_file)
    
    d_parser = DicomParser(dicom_dir, contour_dir, link_file)
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
        plt.savefig('../images/segmentation_vis.png', dpi=300)

    plt.show()
