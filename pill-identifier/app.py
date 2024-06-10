import os
import uuid
from flask import Flask, request, jsonify, render_template
from model import identify_pills, fetch_pill_info
from flask_cors import CORS
import base64

app = Flask(__name__, static_folder='static')

CORS(app) #CORS 설정

#메인 접속점
@app.route('/')
def index():
    return render_template('index.html')

#파일 분석 엔드포인트
@app.route('/identify', methods=['POST'])
def identify():
    if 'front_image' not in request.files or 'back_image' not in request.files:
        return jsonify({'error': '이미지 파일이 업로드되지 않았습니다.'}), 400

    front_image = request.files['front_image']
    back_image = request.files['back_image']

    #다중 접속을 위해 유저 아이디 설정
    user_id = str(uuid.uuid4())
    upload_dir = f'uploads/{user_id}'
    os.makedirs(upload_dir, exist_ok=True)

    front_image_path = f'{upload_dir}/front_{front_image.filename}'
    back_image_path = f'{upload_dir}/back_{back_image.filename}'

    front_image.save(front_image_path)
    back_image.save(back_image_path)

    pill_ids = identify_pills(front_image_path, back_image_path)
    pill_details = fetch_pill_info(pill_ids)

    pill_details['이미지'] = pill_details['품목일련번호'].apply(lambda x: f'static/images/{x}.png')

    os.remove(front_image_path)
    os.remove(back_image_path)
    os.rmdir(upload_dir)

    #딕셔너리화
    return jsonify({'pill_details': pill_details.to_dict(orient='records')})

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)