from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
import cv2
import torch
import numpy as np
import tensorflow as tf
import os
import uuid
import mysql.connector
from ultralytics import YOLO
from datetime import datetime
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from collections import Counter


# Constants
CONFIDENCE_THRESHOLD = 0.7
CHAR_CONFIDENCE_THRESHOLD = 60.0
COLOR = (0, 255, 0)

# Ensure crop save path exists
# os.makedirs(CROP_SAVE_PATH, exist_ok=True)

# Load YOLO models
number_plate_model_path = r"/Users/alishtuladhar/Number/code/runs/detect/train2/weights/best.pt"
# character_segmentation_model_path = r"/Users/alishtuladhar/Number/seg/runs/detect/train2/weights/best.pt"
character_segmentation_model_path = r"/Users/alishtuladhar/Number/character33/weights/best.pt"
classification_model_path = r"/Users/alishtuladhar/Number/cnn/embbmod_1.h5"

number_plate_model = YOLO(number_plate_model_path)
character_segmentation_model = YOLO(character_segmentation_model_path)
classification_model = tf.keras.models.load_model(classification_model_path)

# Define class names for classification
class_names = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection setup
def connect_to_database():
    return mysql.connector.connect(
        host="localhost",
        user="root", 
        password="",  
        database="detection_results"  
    )

# Save detection results to the database
def save_detection_result(file_type: str, file_name: str, recognized_texts: str, detected_characters: str):
    try:
        db = connect_to_database()
        cursor = db.cursor()
        query = """
            INSERT INTO result (file_type, file_name, recognized_texts, detected_characters, detection_time)
            VALUES (%s, %s, %s, %s, NOW())
        """
        cursor.execute(query, (file_type, file_name, recognized_texts, detected_characters))
        db.commit()
        cursor.close()
        db.close()
        print("Data successfully inserted into the database.")
    except mysql.connector.Error as err:
        print(f"Error: {err}")

# Uncommented DB test route
@app.post("/db_test/")
async def db_test(
    file_type: str = Form(...), 
    file_name: str = Form(...), 
    recognized_texts: str = Form(...), 
    detected_characters: str = Form(...)
):
    save_detection_result(file_type, file_name, recognized_texts, detected_characters)
    return {"message": "Data inserted successfully into the database."}

# Fetch all data from the database
def fetch_all_data():
    try:
        db = connect_to_database()
        cursor = db.cursor(dictionary=True)  # Use dictionary=True for row mapping
        query = "SELECT * FROM result"
        cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()
        db.close()
        return data
    except mysql.connector.Error as err:
        print(f"Error fetching data: {err}")
        return []

app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Template configuration
templates = Jinja2Templates(directory="app/templates")

# Character prediction function
def predict_character(image):
    img_height, img_width = 32, 32
    img = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_AREA)
    img_array = np.expand_dims(img, axis=0)
    predictions = classification_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return class_names[np.argmax(score)], 100 * np.max(score)

# Improved sorting logic for characters
def sort_characters(predictions):
    rows = []
    row_threshold = 10

    for char in predictions:
        x1, y1, x2, y2 = char[2]
        placed = False
        for row in rows:
            if abs(row[0][2][1] - y1) < row_threshold:
                row.append(char)
                placed = True
                break
        if not placed:
            rows.append([char])

    sorted_predictions = []
    for row in sorted(rows, key=lambda r: r[0][2][1]):
        sorted_predictions.extend(sorted(row, key=lambda c: c[2][0]))

    return sorted_predictions

# Processing function for images
def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    number_plate_detections = number_plate_model.predict(image)[0].boxes.data

    recognized_texts = []
    detected_characters = []
    if number_plate_detections.shape != torch.Size([0, 6]):
        for detection in number_plate_detections:
            confidence = detection[4]
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            xmin, ymin, xmax, ymax = map(int, detection[:4])
            cropped_image = image[ymin:ymax, xmin:xmax]

            char_detections = character_segmentation_model.predict(cropped_image)[0].boxes.data
            predictions = []

            for char_detection in char_detections:
                char_x1, char_y1, char_x2, char_y2 = map(int, char_detection[:4])
                char_image = cropped_image[char_y1:char_y2, char_x1:char_x2]
                char_label, char_confidence = predict_character(char_image)
                if char_confidence >= CHAR_CONFIDENCE_THRESHOLD:
                    predictions.append((char_label, char_confidence, (char_x1, char_y1, char_x2, char_y2)))

            sorted_predictions = sort_characters(predictions)
            detected_text = "".join([char[0] for char in sorted_predictions])
            recognized_texts.append(detected_text)
            detected_characters.append(detected_text)

            for char_label, char_confidence, (char_x1, char_y1, char_x2, char_y2) in sorted_predictions:
                cv2.rectangle(cropped_image, (char_x1, char_y1), (char_x2, char_y2), COLOR, 1)
                cv2.putText(
                    cropped_image,
                    f"{char_label} ({char_confidence:.1f}%)",
                    (char_x1, char_y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    COLOR,
                    1
                )

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR, 2)
            cv2.putText(
                image,
                f"Number Plate: {detected_text}",
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLOR,
                2
            )

    return image, recognized_texts, detected_characters

# Video Processing Function
def process_video(video_path):
    # Open input video
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    # Get video properties
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Define output file path
    processed_filename = os.path.basename(video_path).replace(".mp4", "-processed.mp4")
    output_path = os.path.join("app", "static", processed_filename)

    # Initialize video writer with H.264 codec for better browser support
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        raise ValueError(f"Failed to create video writer for: {output_path}")

    recognized_texts = []
    frame_count = 0

    # Process each frame
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}...")

        # Process the frame
        try:
            result_frame, texts, _ = process_image(frame)
            recognized_texts.extend(texts)

            # Write the processed frame to the output video
            out.write(result_frame)
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")

        

    # Release resources
    video_capture.release()
    out.release()

    if frame_count == 0:
        raise ValueError("No frames were processed. Check the input video file.")
    
    recognized_texts = [text for text in recognized_texts if text]

# Count occurrences of each item
    text_counts = Counter(recognized_texts)

# Find the maximum count
    max_count = max(text_counts.values())

# Check if all items have a count of 1
    if max_count == 1:
    # Select the first item with length 7
        recognized_texts = [text for text in recognized_texts if len(text) == 7][:1]
    else:
    # Select items with the maximum repetition
        recognized_texts = [text for text, count in text_counts.items() if count == max_count]
    
    for items in recognized_texts:
        print(f"{(items)}")

    print(f"Processed video saved at: {output_path}")
    return output_path, recognized_texts



# Endpoint for the home page
@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

# @app.get("/help", response_class=HTMLResponse)
# async def about(request: Request):
#     return templates.TemplateResponse("help.html", {"request": request})

@app.get("/contact", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})


# Endpoint for image detection
@app.post("/detect_image", response_class=HTMLResponse)
async def detect_number_plate(file: UploadFile = File(...)):
    image_data = await file.read()
    image = np.array(Image.open(BytesIO(image_data)))

    result_image, recognized_texts, detected_characters = process_image(image)
    result_filename = f"{uuid.uuid4()}.jpg".replace("_", "-")
    result_path = os.path.join("app", "static", result_filename)
    cv2.imwrite(result_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

    # Save results to the database
    save_detection_result(
        file_type="image",
        file_name=result_filename,
        recognized_texts=", ".join(recognized_texts),
        detected_characters=", ".join(detected_characters),
    )

    return templates.TemplateResponse(
        "result.html",
        {
            "request": {},
            "image_url": f"/static/{result_filename}",
            "image_detected_character": ", ".join(detected_characters),
            "recognized_texts": recognized_texts,
        },
    )

@app.get("/user", response_class=HTMLResponse)
async def user_data():
    data = fetch_all_data()
    rows = "".join(
        f"""
        <tr>
            <td>{row['id']}</td>
            <td>{row['file_type']}</td>
            <td>{row['file_name']}</td>
            <td>{row['recognized_texts']}</td>
            <td>{row['detected_characters']}</td>
            <td>{row['detection_time']}</td>
        </tr>
        """
        for row in data
    )

    html_content = f"""
    <html>
        <body>
            <h2>Stored Detection Data:</h2>
            <table border="1" style="width: 100%; border-collapse: collapse;">
                <tr>
                    <th>ID</th>
                    <th>File Type</th>
                    <th>File Name</th>
                    <th>Recognized Texts</th>
                    <th>Detected Characters</th>
                    <th>Detection Time</th>
                </tr>
                {rows}
            </table>
            <a href="/"><button>Back</button></a>
            </>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Endpoint for video detection
@app.post("/detect_video", response_class=HTMLResponse)
async def detect_number_plate_from_video(file: UploadFile = File(...)):
    video_data = await file.read()
    video_filename = f"{uuid.uuid4()}.mp4".replace("_", "-")
    video_path = os.path.join("app", "static", video_filename)

    # Save the uploaded video to a temporary path
    with open(video_path, "wb") as f:
        f.write(video_data)

    processed_video_path, recognized_texts = process_video(video_path)

    # Save results to the database
    save_detection_result(
        file_type="video",
        file_name=os.path.basename(processed_video_path),
        recognized_texts=", ".join(recognized_texts),
        detected_characters=", ".join(recognized_texts),
    )

    return templates.TemplateResponse(
        "result.html",
        {
            "request": {},
            "video_url": f"/static/{os.path.basename(processed_video_path)}",
            "video_detected_character": ", ".join(recognized_texts),
            "recognized_texts": recognized_texts,
        },
    )
