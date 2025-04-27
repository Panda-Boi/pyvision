# pyvision

## Code
Each significant section of the entire system is divided into its own class with their relevant methods.
The configuration is done using the config class which is initalized and passed into the Top level DetectionSystem class which processes the video.

## Running
To run the project simply supply an input.mp4 file (or use the provided input) and run

`python -m venv .venv`  
`pip install -r requirements.txt`  
`python main.py`

## Division of work
Prerna Saxena - Person Segmentation using YOLO  

Dhruv Sharma - Batch Processing, Helmet Detection using custom trained model, Linear Frame Interpolation  

Nihar Puthin - Pose analysis using Google movenet, Working state analysis by a linear differential of pose keypoints, Deepsort for person tracking between frames
