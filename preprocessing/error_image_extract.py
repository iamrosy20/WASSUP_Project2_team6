import pandas as pd
import requests
from tqdm import tqdm

df = pd.read_csv('data/drug_data.csv')

with open('data/error.txt', 'r') as file:
    numbers = file.readlines()
    numbers = [int(number.strip()) for number in numbers]

df2 = df[df['품목일련번호'].isin(numbers)]

error_drug = []

for index, row in tqdm(df2.iterrows(), total=df2.shape[0]):
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
