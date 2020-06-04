SUMMARY
This document is a report for BBM-418 Computer Vision Laboratory Assignment-1. An implementation of Hough Transform for line detection was done and results are given for this assignment. Sample space consists of bar-code images which has different angles and visibility. Python is used as programming language and OpenCV is used for processing library. Results show that we are able to detect bar-codes with some errors and missing parts.
You can check report file for more information.

HOW TO RUN THE CODE:
To run this project you need to have numpy and opencv packages on your computer. After installing packages follow the given steps in order.

1) You need to name "original subset" and "detection subset" image pairs with same number starting from 1 up to 14.
2) Open "Utils.py" folder.
3) Set "original_subset_path" and "detection_subset_path" variables to where you store the images. 
(
 For example original_subset_path = r"C:\Users\AliIhsan\Desktop\Assignment_1\Dataset\Original_Subset\\"
	      detection_subset_path = r"C:\Users\AliIhsan\Desktop\Assignment_1\Dataset\Detection_Subset\\"
)
4) Save and close "Utils.py" folder.
5) Open command prompt or Anaconda Prompt and navigate to "code" folder
6) Type 'python main.py" and hit enter.
7) You start seeing logs if it's working



