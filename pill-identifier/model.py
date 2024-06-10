import cv2
import numpy as np
from rembg import remove
import tensorflow as tf
import pandas as pd
from difflib import SequenceMatcher
import base64
import requests
from Levenshtein import distance as levenshtein_distance
from math import floor

SHAPE_MODEL_PATH = 'models/cnn_shape_final.h5'
COLOR_MODEL_PATH = 'models/cnn_color_left_final.h5'
CSV_FILE_PATH = 'data/dataframe.csv'

df = pd.read_csv(CSV_FILE_PATH)

def process_image(image_path):
    # 이미지 로드
    img = cv2.imread(image_path)
    
    # rembg를 사용하여 배경 제거
    img_nobg = remove(img)

    # rembg는 알파 채널을 추가하므로 RGB 이미지로 변환
    img_nobg = cv2.cvtColor(img_nobg, cv2.COLOR_BGRA2BGR)
    
    # 이미지 크기 확인
    h, w = img_nobg.shape[:2]

    # 타겟 비율 계산 (7:13)
    target_ratio = 7 / 13

    # 중앙에 배치하면서 7:13 비율로 크롭
    if w / h > target_ratio:
        # 너비가 비율보다 큰 경우, 너비를 조정
        new_w = int(h * target_ratio)
        start_w = (w - new_w) // 2
        cropped_img = img_nobg[:, start_w:start_w + new_w]
    else:
        # 높이가 비율보다 큰 경우, 높이를 조정
        new_h = int(w / target_ratio)
        start_h = (h - new_h) // 2
        cropped_img = img_nobg[start_h:start_h + new_h, :]

    # 최종 이미지를 70x130으로 리사이즈
    resized_img = cv2.resize(cropped_img, (70, 130), interpolation=cv2.INTER_AREA)

    return resized_img

#배경제거
def remove_background(image_path):
    input_image = cv2.imread(image_path)
    output_image = remove(input_image)
    cv2.imwrite("no_bg.png", output_image)
    return "no_bg.png"

#형태 예측
def predict_shape(image_path):
    model = tf.keras.models.load_model(SHAPE_MODEL_PATH)
    image = process_image(image_path)
    image = image / 255.0
    predictions = model.predict(tf.expand_dims(image, 0))
    return predictions.argmax(axis=1)[0]

#색상 예측
def predict_color(image_path):
    model = tf.keras.models.load_model(COLOR_MODEL_PATH)
    image = process_image(image_path)
    image = image / 255.0
    predictions = model.predict(tf.expand_dims(image, 0))
    return predictions.argsort(axis=1)[0][::-1][:3]

#색상과 형태 비교
def compare_shapes_and_colors(shape1, shape2, colors1, colors2):
    if shape1 != shape2:
        return False, "Shape mismatch."
    if colors1[0] != colors2[0]:
        if colors1[0] not in colors2[:3] and colors2[0] not in colors1[:3]:
            return False, "Color mismatch."
    return True, ""

# 이미지 인코딩 (GPT가 인식하기 위해 쓰임)
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

#OPEN AI API 불러 오는 코드
def extract_text(image_path):
    base64_image = encode_image(image_path)
    api_key = "api_key" #실제 KEY를 사용해야함

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please print out the letters written on the pill. If the recognition fails, print out 'zzzzzz'. Just tell me the only text. You should never add another additional explanation."
                    }, #AI가 알아듣게 하는 프롬프트
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

    if response.status_code == 200 and 'choices' in response.json():
        text_content = response.json()['choices'][0]['message']['content']
        print(f"Extracted Text: {text_content}")
        return str(text_content)  
    else:
        print("Failed to extract text or API call was unsuccessful.")
        return ""  


def calculate_similarity(text, reference):
    # 양쪽을 string으로 만듦
    text = str(text)
    reference = str(reference)
    
    max_length = max(len(text), len(reference))
    if max_length == 0:
        return 1  #양쪽이 비었을경우 1을 배출함
    
    # 유사도 검사, 레벤슈타인
    similarity = (max_length - levenshtein_distance(text, reference)) / max_length
    return min(similarity, 1)

#최종 함수
def identify_pills(front_image_path, back_image_path):
    process_info = {}
    for label, image_path in [('front', front_image_path), ('back', back_image_path)]:
        no_bg = remove_background(image_path)
        shape = predict_shape(no_bg)
        colors = predict_color(no_bg)
        text = extract_text(no_bg)
        process_info[label] = (shape, colors, text if text is not None else "")

    text_matches = []
    for front_label, back_label in [('front', 'back'), ('back', 'front')]: #앞 뒤가 바뀌어도 상관없게 처리하는 부분
        text_front, text_back = process_info[front_label][2], process_info[back_label][2]
        df_filtered = df.assign(
            front_similarity=df.apply(lambda row: calculate_similarity(text_front, row['front']), axis=1),
            back_similarity=df.apply(lambda row: calculate_similarity(text_back, row['back']), axis=1)
        )
        text_matches.extend(df_filtered[
            (df_filtered['front_similarity'] >= 0.9) & (df_filtered['back_similarity'] >= 0.9)
        ]['id'].tolist())

        #텍스트 신뢰도가 높으면 그냥 값을 반환함

    #텍스트를 신뢰할 수 없을때, 가중치를 줘서 모양, 색상, 텍스트를 비교함
    if text_matches:
        return text_matches[:5]
    else:
        all_matches = []
        selected_ids = set()
        for front_label, back_label in [('front', 'back'), ('back', 'front')]:
            shape_front, colors_front, _ = process_info[front_label]
            shape_back, colors_back, _ = process_info[back_label]
            text_front, text_back = process_info[front_label][2], process_info[back_label][2]

            df_filtered = df.assign(
                front_similarity=df.apply(lambda row: calculate_similarity(text_front, row['front']), axis=1),
                back_similarity=df.apply(lambda row: calculate_similarity(text_back, row['back']), axis=1),
                color_similarity=df.apply(lambda row: 1 if any(color in [str(row['color'])] for color in colors_front) else 0, axis=1),
                shape_similarity=df.apply(lambda row: 1 if row['shape'] == shape_front else 0, axis=1)
            )

            df_filtered['text_similarity'] = (df_filtered['front_similarity'] + df_filtered['back_similarity']) / 2
            df_filtered['color_weight'] = 1 - (df_filtered['text_similarity'] - 0.5) / 0.4
            df_filtered['color_weight'] = df_filtered['color_weight'].clip(0, 1)
            df_filtered['shape_weight'] = 1 - (df_filtered['text_similarity'] - 0.5) / 0.4
            df_filtered['shape_weight'] = df_filtered['shape_weight'].clip(0, 1)

            df_filtered['total_similarity'] = (
                0.5 * df_filtered['text_similarity'] +
                0.1 * df_filtered['color_weight'] * df_filtered['color_similarity'] +
                0.4 * df_filtered['shape_weight'] * df_filtered['shape_similarity']
            )

            # 중복 제거
            df_filtered = df_filtered[~df_filtered['id'].isin(selected_ids)]

            top_matches = df_filtered.nlargest(5, 'total_similarity')
            all_matches.extend(top_matches['id'].tolist())
            selected_ids.update(top_matches['id'].tolist())

        if all_matches:
            return all_matches[:5]
        else:
            return []

# 웹에서 정보를 표시하기 위한 함수
def fetch_pill_info(pill_ids):
    info_df = pd.read_csv('data/info.csv')
    pill_info = info_df[info_df['품목일련번호'].isin(pill_ids)]
    
    # NaN 값을 '정보 없음'으로 대체
    pill_info = pill_info.fillna('정보 없음')
    
    return pill_info[['품목일련번호', '품목명', '효능효과', '용법용량', '주의사항']]