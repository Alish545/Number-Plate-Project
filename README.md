# **Vehicle Number Plate Detection and Character Recognition**
## Overview
This project implements an Automatic Number Plate Recognition (ANPR) system using YOLOv8 for number plate detection and a custom CNN model for character recognition. The system is integrated into a web application using FastAPI, allowing users to upload images or videos and obtain recognized plate numbers in real-time.

## Features
Number Plate Detection: Uses YOLOv8 to detect license plates from images/videos.
Character Segmentation & Recognition: Segments and recognizes characters using a CNN model.
Web Interface: Built using FastAPI for easy interaction.
Database Storage: Stores recognized plates in a MySQL database.
High Accuracy: Achieves up to 92% accuracy under ideal conditions.
Technologies Used
Machine Learning: YOLOv8, CNN (TensorFlow)
Computer Vision: OpenCV
Backend: FastAPI
Database: MySQL
Frontend: HTML, CSS

## API Endpoints
Endpoint	Method	Description
/upload	POST	Uploads an image/video for number plate detection.
/detect_image	POST	Detects and recognizes number plate characters in an image.
/detect_video	POST	Processes video frames for plate detection.
/results	GET	Retrieves previous recognition results from the database.

## Future Improvements
Real-time video processing.
Handling different lighting conditions and plate styles.
Improving recognition accuracy with additional training.

## Authors
Alish Tuladhar
Arbin Shrestha
Shreejan Shrestha
