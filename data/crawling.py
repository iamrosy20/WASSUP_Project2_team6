import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm
import base64
import numpy as np

df = pd.read_csv('/home/kdt-admin/data/drug_data_2.csv')

new_df = pd.DataFrame(columns=['품목일련번호', '상품명', '효능효과', '용법용량', '주의사항'])

if not os.path.exists('df'):
    os.makedirs('df')

failed_codes = []

# 진행률 표시를 위해 tqdm 사용
for code in tqdm(df['품목일련번호'], desc='크롤링 진행률'):
    try:
        url = f'https://nedrug.mfds.go.kr/pbp/CCBBB01/getItemDetailCache?cacheSeq={code}aupdateTs2023-12-22%2017:06:40.158115b'
        response = requests.get(url, timeout=10)  # 요청 시간 초과를 위한 timeout 추가
        soup = BeautifulSoup(response.text, 'html.parser')

    except requests.RequestException:
        print(f"품목일련번호 {code}에 대한 요청 실패")
        failed_codes.append(code)  # 실패한 코드 저장
        continue  # 실패시 다음 코드로 계속
    
    # 한글상품명 이동
    try:
        product_name = df.loc[df['품목일련번호'] == code, '품목명'].values[0]
    except IndexError:
        product_name = np.nan

    # 효능효과 추출
    try:
        efficacy = '\n'.join([p.text.strip() for p in soup.select('#_ee_doc > p')])
    except AttributeError:
        efficacy = np.nan
    
    # 용법용량 추출
    try:
        dosage = '\n'.join([p.text.strip() for p in soup.select('#_ud_doc > p')])
    except Exception:
        dosage = np.nan
    
    # 주의사항 추출
    try:
        precautions = '\n'.join([p.text.strip() for p in soup.select('#_nb_doc > p')])
    except Exception:
        precautions = np.nan
        
    # 새로운 DataFrame에 저장
    data_to_add = {
        '품목일련번호': [code],
        '품목명': [product_name],
        '효능효과': [efficacy],
        '용법용량': [dosage],
        '주의사항': [precautions]
    }

    new_df2 = pd.DataFrame(data_to_add)
    new_df = pd.concat([new_df, new_df2], ignore_index=True)

new_df.to_csv('result.csv', index=False)

with open('failed_codes.txt', 'w') as f:
    for code in failed_codes:
        f.write(f"{code}\n")