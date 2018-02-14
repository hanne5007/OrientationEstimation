“””
README file for 
Project 2 Orientation Estimation
by Hantian Liu
“””


Run displayResults.py to run UKF for IMU data, and show plots of Euler angles, in comparison with ground truth, and then show the panorama, for each dataset in the data folder.
Videos of generating panorama would be saved in local folder, whose name shows the data source, e.g. 'VideoIMU1.mp4', 'VideoVicon9.mp4'.  

## Folders
Change the folder name, 'camfolder' for images, 'imufolder' for IMU data, 'viconfolder' for Vicon data. 
'max_dataset_num' is the largest dataset number to test, now equals to 9 as in training se, needs to be changed later.

## Orientation Plots
If the orientation plots for Euler angles are not needed, set 'show_rpy_plots' to False. Here 'show_imu_only_rpy_plots' means whether to show the orientation estimation from results only based on gyro or accelerometer, in order to show the effectiveness of UKF in comparison. 

## Panorama Plots
While if panorama is not needed, set 'show_panorama' to False. Here 'show_panorama_from_vicon' refers to whether or not generating the videos from Vicon data.
Since there are frames at the beginning and the end of videos, which are still and would not help the final panorama evaluation, I set 'first_still' as frames to start stitching panoramas, and 'last_still' as numbers of frames before the last frame to end stitching the panoramas. 

## Ground Truth
If there does not exist ground truth for data, 'vicon_exists' needs to be set to False. However, if there does not exist corresponding image files, my algorithm can automatically pass the panorama part for that dataset, like what we have in the training set. 




