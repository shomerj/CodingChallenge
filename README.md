# Coding Challenge

All code written for this challenge will be located in the 'src' directory. You will find four different scripts that I created and the additional parsing.py script that was provided. The 'image' directory has some sample images taken from program logs and data visualization

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
