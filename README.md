## 프로젝트 소개

### 이미지를 통한 알약 분석
-----

시간이 갈 수록 약물의 과대처방이 심각해지고 있고 이로 인한 약물의 오남용이 증가하고 있는 실정이다. 이를 방지하기 위해 스마트폰으로 알약을 찍으면 그것이 무슨 알약인지 알려주는 서비스를 제공하고자 프로젝트를 시작하였다.

## 🕘 프로젝트 기간
**START  : 2024.03.27**
<br>
**END : 2024.04.15**

## 🧑‍💻 팀 구성
- **김수현** - 프로젝트 구상, EDA, 데이터 전처리, CNN 모델링
- **이구협** - 조장, 발표, 프로젝트 구상, EDA, 데이터 전처리, OCR 모델링, 서비스 구축
- **최한솔** - 프로젝트 구상, EDA, 데이터 전처리
- **황승욱** - 프로젝트 구상, EDA, 데이터 전처리

## ⌨ 개발 환경
### Language
------
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/javascript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black">

### OS
------
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)

> ESTSoft에서 서버를 제공하여 다양한 작업을 우분투 환경에서 진행하였음.

### IDE 
------
<img src="https://img.shields.io/badge/ide-%23121011?style=for-the-badge"><img src="https://img.shields.io/badge/visual studio code-007ACC?style=for-the-badge&logo=visual studio code&logoColor=white"><br>
<img src="https://img.shields.io/badge/ide-%23121011?style=for-the-badge"><img src="https://img.shields.io/badge/jupyter-F37626?style=for-the-badge&logo=Jupyter&logoColor=white"> 


## 전체적인 모델링 프로세스
### 데이터 분석 및 EDA, 전처리
------

#### 데이터 추출

[의약품 안전나라]] (https://nedrug.mfds.go.kr/pbp/CCBGA01/getItem?totalPages=4&limit=10&page=2&&openDataInfoSeq=11)

> 해당 데이터셋을 받아 활용하였음

> 25,125개의 약품 데이터가 30개의 column으로 구분되어있었음

- 품목일련번호
- 품목명
- 업소일련번호
- 업소명
- 성상
- 큰제품이미지
- 표시앞
- 표시뒤
- 의약품제형
- 색상앞
- 색상뒤
- 분할선앞
- 분할선뒤
- 크기장축
- 크기단축
- 크기두께
- 이미지생성일자
- 분류번호
- 분류명
- 전문일반구분
- 품목허가일자
- 제형코드명
- 표기내용앞
- 표기내용뒤
- 표기이미지앞
- 표기이미지뒤
- 표기코드앞
- 표기코드뒤
- 변경일자
- 사업자번호


#### 이상치 제거

<img width="575" alt="image" src="https://github.com/iamrosy20/WASSUP_Project2_team6/assets/160453988/971ed275-bb90-49c7-91d0-8b6886e4d28e">

> 이상치가 너무 많으면 column 채로 제거하였고, 색상이나 모양은 직접 이미지를 보고 처리하였음.

#### 알약 사진 추출

```
df = pd.read_csv('data/drug_data.csv')

error_drug= [] #리스트를 초기화합니다

#tqdm 적용
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    image_url = row['큰제품이미지']
    item_serial = row['품목일련번호']

    
    if pd.notnull(image_url) and pd.notnull(item_serial):
        try:
            response = requests.get(image_url)

            #접속에 성공한경우
            if response.status_code == 200:
                
                # 이미지 파일로 저장
                with open(f'data/images/{item_serial}.png', 'wb') as file:
                    file.write(response.content)

        #오류가 난 경우
        except Exception as e:
            print(f'Error downloading {image_url}: {e}')
            #오류가 발생한 항목을 리스트에 추가
            error_drug.append(item_serial)

#에러가 발생한 항목만 따로 txt파일로 저장합니다.
with open('data/error.txt', 'w') as f:
    for item_serial in error_drug:
        f.write(f'{item_serial}\n')
```

> csv파일 '큰제품이미지' column에 있는 각 row에서 이미지 URL을 통해 이미지를 다운로드하였음


#### Rembg를 통해 배경 제거

> 다양한 방법을 시도해보았으나, Rembg를 통해 배경을 제거하는 것이 개발 시간과 비용의 Tradeoff를 고려한 결과 최적이라고 판단하여 이를 사용하였음

#### 알약 이미지 앞뒤 분리

<img width="546" alt="image" src="https://github.com/iamrosy20/WASSUP_Project2_team6/assets/160453988/922d2812-0c12-4bcc-8bde-8be013c74e91">

> 알약의 이미지를 앞뒤로 분리하여 이미지 인식률을 높힘

### 모델링

#### 이미지 예측 모델

<img width="586" alt="image" src="https://github.com/iamrosy20/WASSUP_Project2_team6/assets/160453988/d47fc749-f816-455a-b9cc-64824f212c1a">


```
images = [image.load_img(p, target_size=(130, 70))   # 780, 420
          for p in glob('data/shape/circle/*png') + glob('data/shape/rectangle/*png') + glob('data/shape/ellipse/*png') + glob('data/shape/square/*png')
          + glob('data/shape/octagon/*png') + glob('data/shape/triangle/*png') + glob('data/shape/rhombus/*png') + glob('data/shape/pentagon/*png')
          + glob('data/shape/hexagon/*png') + glob('data/shape/semicircle/*png') + glob('data/shape/etc/*png')]
image_vector = np.asarray([image.img_to_array(img) for img in images])

# Set labels
y = [10] * 9706 + [9] * 6948 + [8] * 6666 + [7] * 276 + [6] * 274 + [5] * 235 + [4] * 91 + [3] * 58 + [2] * 50 + [1] * 3 + [0] * 479

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(image_vector, y, test_size=0.20, random_state=42)

# Scale the input data
X_train, X_test = X_train / 255, X_test / 255

# Convert to categorical
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

# Build model
def build(input_shape, classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(classes, activation='softmax'))
    return model

# Initialize and fit the model
model = build((130, 70, 3), 11)
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(learning_rate=0.001), metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.2)

```


> 모델은 기본적으로 CNN을 사용하였고, Layer를 조정하는 방식으로 결과를 업데이트함. 알약의 색상과 모양을 유추해냄

#### OCR

> EasyOCR, Tesseract를 사용하여 모델링을 시도해보았으나, 결국 Open AI API를 사용하였음. 알약에 적힌 글자를 유추해냄


#### 종합모델

> 먼저 알약을 인식하면 CNN모델을 통해서 모양과 색상을 유추해냄, 그 다음 API가 예측한 알약에 적힌 글자를 레벤슈타인거리(Levenshtein Distance)를 통해 가장 인접한 글자와 비교해서 정답데이터를 유추해냄
>
> 이것이 가능한것은, 한국에서 허가받은 알약 데이터를 전부 적혀있는 데이터가 존재하기 때문.
>
> 일종의 유일성 정리

## 서비스 
### 서비스 개요
<img width="200" alt="image" src="https://github.com/iamrosy20/WASSUP_Project2_team6/assets/160453988/04b909e7-6dc8-4c21-b9bf-b36fe46cd78b">

> 메인화면

<img width="200" alt="image" src="https://github.com/iamrosy20/WASSUP_Project2_team6/assets/160453988/d0f5a212-c974-4e0d-9c6f-be2ceb0ee776">

> 촬영을 시작하면 이미지 사진을 서버로 보냄

<img width="200" alt="image" src="https://github.com/iamrosy20/WASSUP_Project2_team6/assets/160453988/4a4c791d-3a0f-4cc6-b558-195c83d4ab7b">

> 인식이 불명확할 경우, 가장 비슷하다고 모델이 판단한 알약 5개를 표시함

<img width="200" alt="image" src="https://github.com/iamrosy20/WASSUP_Project2_team6/assets/160453988/d50f223a-4dbb-495b-a0bb-92b1475f7b47">

> 알약의 이름을 클릭하면 의약품 안전나라에서 크롤링한 용법과 부작용 등의 자료를 표시함


### API 프롬프트

```
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
```

> 알약에 글자가 적혀있지 않은 경우 zzzzzz를 출력해서 문자열 유사도를 다른 글자에서 멀게 만듦

### 서버 구조

```
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
```

> Flask로 구현하였음