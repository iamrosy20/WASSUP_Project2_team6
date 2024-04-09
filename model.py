import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from rembg import remove
import tensorflow as tf
import pandas as pd
from difflib import SequenceMatcher
from openai import OpenAI
import base64
import requests

SHAPE_MODEL_PATH = 'cnn_shape.h5'
COLOR_MODEL_PATH = 'cnn_color.h5'
CSV_FILE_PATH = 'dataframe.csv'

df = pd.read_csv(CSV_FILE_PATH)
image_path = ""

def resize_image(image_path, size):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, size)
    return resized_image

def remove_background(image_path):
    input_image = cv2.imread(image_path)
    output_image = remove(input_image)
    cv2.imwrite("no_bg.png", output_image)
    return "no_bg.png"
    
def predict_shape(image_path):
    model = tf.keras.models.load_model(SHAPE_MODEL_PATH)
    image = resize_image(image_path, (70, 130))
    image = image / 255.0
    predictions = model.predict(tf.expand_dims(image, 0))
    shape_index = predictions.argmax(axis=1)[0]
    confidence = predictions[0][shape_index]
    return shape_index, confidence * 10

def predict_color(image_path):
    model = tf.keras.models.load_model(COLOR_MODEL_PATH)
    image = resize_image(image_path, (70, 130))
    image = image / 255.0
    predictions = model.predict(tf.expand_dims(image, 0))
    color_index = predictions.argmax(axis=1)[0]
    confidence = predictions[0][color_index]
    return color_index, confidence * 10

def compare_with_csv(shape, color):
    matched = df[(df['shape'] == shape) & (df['color'] == color)]
    return matched

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def final_identification(image_path, matched_df):
    base64_image = encode_image(image_path)
    
    api_key = "YOUR_API_KEY"

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Please print out the letters written on the pill. Just tell me the only text."
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    extracted_texts = response.json()['choices'][0]['message']['content']

    best_match = None
    highest_similarity = 0

    for _, row in matched_df.iterrows():
        front_similarity = similar(extracted_texts, str(row['front']))
        back_similarity = similar(extracted_texts, str(row['back']))
        max_similarity = max(front_similarity, back_similarity)

        if max_similarity > highest_similarity:
            highest_similarity = max_similarity
            best_match = row

    if highest_similarity >= 0.9:
        return best_match['name'], highest_similarity * 10
    else:
        shape, shape_score = predict_shape(image_path)
        color, color_score = predict_color(image_path)
        
        if shape_score >= 8 and color_score >= 8:
            matched_df = matched_df[(matched_df['shape'] == shape) & (matched_df['color'] == color)]
            if not matched_df.empty:
                return matched_df.iloc[0]['name'], min(shape_score, color_score)
        
        return None, max(highest_similarity * 10, shape_score, color_score)


def update_gui_with_results(result):
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, result)

def identify_image():
    global image_path
    if image_path:
        image_path_no_bg = remove_background(image_path)

        identification_result, total_score = final_identification(image_path_no_bg, df)

        if identification_result:
            result = f"분석 결과: 이 약은 [{identification_result}] 입니다."
            if total_score >= 8:
                confidence = "높음"
            elif total_score >= 6:
                confidence = "중간"
            else:
                confidence = "낮음"
            result += f" (신뢰도: {confidence})"
        else:
            result = "이미지를 명확하게 식별하지 못했습니다."
            if total_score >= 8:
                confidence = "높음"
            elif total_score >= 6:
                confidence = "중간"
            else:
                confidence = "낮음"
            result += f" (유사도: {confidence})"

        update_gui_with_results(result)
    else:
        print("이미지를 식별하지 못했습니다.")

def open_and_display_image():
    global image_path
    file_path = filedialog.askopenfilename()
    if file_path:
        image_path = file_path
        img = Image.open(file_path)
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        image_label.configure(image=img)
        image_label.image = img


root = tk.Tk()
root.title("알약 식별")

image_frame = tk.Frame(root)
image_frame.pack(fill=tk.BOTH, expand=True)

image_label = tk.Label(image_frame)
image_label.pack()

result_text = tk.Text(root, height=10, width=50, wrap=tk.WORD)
result_text.pack(fill=tk.BOTH, expand=True)

open_btn = tk.Button(root, text="이미지 업로드", command=open_and_display_image)
open_btn.pack()

identify_btn = tk.Button(root, text="알약 식별 시작", command=identify_image)
identify_btn.pack(pady=10)


root.mainloop()