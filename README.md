# Coding Challenge Phase 1

All code written for this challenge will be located in the 'src' directory. You will find four different scripts that I created and the additional parsing.py script that was provided. The 'images' directory has some sample images taken from program logs and data visualization

Script Descriptions:

1. dicom_parser.py

   The functionality of this script is to filter out all DICOM files that do not have any associated contour files and return a csv file that links each DICOM file with its associated contour file.

2. parsing_unittest.py

   This is a unit test to ensure the validity of the previously described parser. In this test I am ensuring that the number of contour files are the same as the number of rows in the outputted csv file. I am also ensuring that the linking between the DICOM and contour file is correct.

3. training_pipeline.py

   In training_pipeline.py a class called TrainingPipeline has been created in order to take the csv file previously describe and iterate through the data returning a numpy array of images and targets of a certain batch size.

4. visualize_data.py

   This is a data visualization tool that will display random images, there associated masks, and the segmented image.  



### PART 1: Parse the DICOM images and Contour Files

<b>Question 1:</b> There were three main ways I verified that I was parsing the contours correctly. The first was to create a unittest (this is the test described above). Secondly, I incorporated a 'DEBUG' tag in my code that if set to True would display print statements that allowed me to see what my code was producing. There is a screen shot of the log in the images directory. Lastly, I created a data visualization tool (also described above) to ensure that the contours were linked to the correct DICOM image.

<b>Question 2:</b> I incorporated some command line arguments so that with a few flags the code can be run easily.  


### PART 2: Model training pipeline

<b>Question 1:</b> Both pipeline for part 1 and part 2 are fairly distinct processes. This is something that I believe I would change in the future. The functionality of part 1 was to organize the data in a way, such that, the data was coherent and ready to be passed to my training pipeline.  The training pipeline is a generator that loads one batch at a time into memory and then can be passed to the model.

<b>Question 2:</b> I verified that it was working correctly by administering print statements throughout the process. I also displayed the images returned from the script. This was not document in the code.

<b>Question 3:</b> I can envision one possible issue in the future. In this process I am assuming that all images are of the same size, which in this case is true. But if images were introduce other than a resolution of 256^2 the code would break.  There are a few enhancements that I would build into the pipeline given more time and a little bit more domain knowledge. If data augmentation is needed that would be built into threading.  



# Coding Challenge Phase 2

The code for the second phase of the take-home challenge is located in the src directory. There is a new completely new python file called ```LV_segmentation.py```. This files contains the code for exploring different thresholds and types of visualization in an attempt to segment the heart muscle from inside the blood pool. I also add to both the ```visualize_data.py``` and the ```dicom_parer.py``` in order to accommodate the new task of parsing the o-contours. I will go into more detail about all of the above in the next sections.


Script Descriptions:

   1. dicom_parser.py

      This script has the same functionality as in phase 1. The only difference is that I incorporated the parsing of the o-contours as well.

   2. visualize_data.py

      A new function was added to this script so that I could visualize the borders of both the inner and outer contours. Visualizing the contours was necessary to get a better understanding of the data. You can view the images produced from the function ```draw_contours ``` in the ```images/contours/``` directory.

   3. LV_segmentation.py

      This python files has the bulk of the image segmentation exploration. The goal of this file is to determine if simple thresholding is an appropriate method of segmenting the data. There are three functions within the file.

      - ```compare_pixel_intensities```: This function was used to determine the difference in pixel intensities between the heart muscle and the blood pool. You can see a sample of the saved figures from this function in the ```images/pixel_intensity/``` directory.

      - ```lv_segmentation```: This function was used as a visual guide to determine the optimal threshold value. The function returned images displaying the effect of different threshold values. Around each of the inner contour the original outer contour circle was drawn as a reference. You can view the figures from this function in the ```images/segmentation/``` directory.

      - ```contours_from_thresholds```: This functions used the threshold from the previous function and draw contours based on them. The contours returned from this function would be the inner contours desired for this segmentation. You can view the figures from this function in the ```images/thresholds_contours/``` directory.


### Part 1: Parse the o-contours

As describe a little bit above, there was a small adjustment made to the ```dicom_parser.py``` script in order to also parse the o-contours. You just have to pass, into the class, the contour type to create an inner contour or outer contour csv file. In order prep the data to train you just pass the desired csv file into the training pipeline.

### Part 2: Heuristic LV Segmentation approaches

<b>Question 1:</b> After spending a little bit of time on this question there may be a way, with some more fine tuning, to properly segment the inner contours. This is a bit of a tricky question without knowing the expectation and quality of this segmentation. By using a simple heuristics like this, there will be some error in the contours due to the fact that one threshold value may not suffice for the variance in images.

<b>Question 2:</b> I think that exploring different heuristics would be a good idea. Canny edge detection would be an approach I would try.

<b>Question 3:</b> An image segmentation deep learning model, such as, Masked-RCNN would be a good approach for this job.

<b>Question 4:</b> The advantages of using a deep learning segmentation model is that the model can generalize much better than a simple heuristic approach. The ability for a deep learning model to generalize requires a lot labeled data and model tuning in order to get the desired results. Needing to label and train the model with many sample images would be one disadvantage to this approach. Another disadvantage would be inference time. Unlike a heuristic approach, segmenting the image is much slower using a deep learning model, how much slower is dependent on the machine you are using.


### Extra: Possible Errors in Labels

After view some of the given outer and inner contours it seems like there may be some error in the annotations. Without much domain knowledge I cannot be sure, but it would be worth looking into before training a model or moving forward with automating the segmentation process. Here are two example of reasons why there may be an error:

![contour image]('https://github.com/shomerj/CodingChallenge/blob/master/images/contours/contour_img43.png')

![intensity image]('https://github.com/shomerj/CodingChallenge/blob/master/images/pixel_intensity/pixel_intensity_15.png')
