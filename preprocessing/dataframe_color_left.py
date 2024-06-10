import os
import pandas as pd
import shutil

# Read every file name
file_list = os.listdir('data/modified_images/RL')

# Split file name and its extension
file_name = []
for one_file_list in file_list:
  if 'left' in one_file_list:
    if '.png' in one_file_list:
      name = one_file_list.split('.')[0]
      file_name.append(name[5:])

# Load drug_data.csv
df = pd.read_csv('data/drug_data.csv')

# Replace - value in front color with yellow color
df.replace({'색상앞': '-'}, '노랑', inplace=True)

# Slice front color value based on the dividing line |
df['색상앞'] = df['색상앞'].str.slice(start=0, stop=2)

# Check color of pill and copy image into appropriate folder
for one_file in file_name:
  source = f'data/modified_images/RL/left_{one_file}.png'

  index_num = df[df['품목일련번호'] == int(one_file)].index
  if df.iloc[index_num[0]]['색상앞'] == '하양':         # white 9664
    destination = f'data/color_front/white/{one_file}.png'
  elif df.iloc[index_num[0]]['색상앞'] == '분홍':       # pink 3472
    destination = f'data/color_front/pink/{one_file}.png'
  elif df.iloc[index_num[0]]['색상앞'] == '노랑':       # yellow 3145
    destination = f'data/color_front/yellow/{one_file}.png'
  elif df.iloc[index_num[0]]['색상앞'] == '주황':       # orange 2326
    destination = f'data/color_front/orange/{one_file}.png'
  elif df.iloc[index_num[0]]['색상앞'] == '갈색':       # brown 1538
    destination = f'data/color_front/brown/{one_file}.png'
  elif df.iloc[index_num[0]]['색상앞'] == '파랑':       # blue 1452
    destination = f'data/color_front/blue/{one_file}.png'
  elif df.iloc[index_num[0]]['색상앞'] == '연두':       # lightgreen 1245
    destination = f'data/color_front/lightgreen/{one_file}.png'
  elif df.iloc[index_num[0]]['색상앞'] == '초록':       # green 1024
    destination = f'data/color_front/green/{one_file}.png'
  elif df.iloc[index_num[0]]['색상앞'] == '빨강':       # red 695
    destination = f'data/color_front/red/{one_file}.png'
  elif df.iloc[index_num[0]]['색상앞'] == '회색':       # gray 197
    destination = f'data/color_front/gray/{one_file}.png'
  elif df.iloc[index_num[0]]['색상앞'] == '보라':       # purple 139
    destination = f'data/color_front/purple/{one_file}.png'
  elif df.iloc[index_num[0]]['색상앞'] == '청록':       # cyan 112
    destination = f'data/color_front/cyan/{one_file}.png'
  elif df.iloc[index_num[0]]['색상앞'] == '검정':       # black 73
    destination = f'data/color_front/black/{one_file}.png'
  elif df.iloc[index_num[0]]['색상앞'] == '자주':       # violet 65
    destination = f'data/color_front/violet/{one_file}.png'
  elif df.iloc[index_num[0]]['색상앞'] == '남색':       # navy 43
    destination = f'data/color_front/navy/{one_file}.png'
  else:                                                 # transparent 25
    destination = f'data/color_front/transparent/{one_file}.png'

  shutil.copyfile(source, destination)

white_list = os.listdir('data/color_front/white')
pink_list = os.listdir('data/color_front/pink')
yellow_list = os.listdir('data/color_front/yellow')
orange_list = os.listdir('data/color_front/orange')
brown_list = os.listdir('data/color_front/brown')
blue_list = os.listdir('data/color_front/blue')
lightgreen_list = os.listdir('data/color_front/lightgreen')
green_list = os.listdir('data/color_front/green')
red_list = os.listdir('data/color_front/red')
gray_list = os.listdir('data/color_front/gray')
purple_list = os.listdir('data/color_front/purple')
cyan_list = os.listdir('data/color_front/cyan')
black_list = os.listdir('data/color_front/black')
violet_list = os.listdir('data/color_front/violet')
navy_list = os.listdir('data/color_front/navy')
transparent_list = os.listdir('data/color_front/transparent')

print(len(white_list))          # 9482
print(len(pink_list))           # 3445
print(len(yellow_list))         # 3079
print(len(orange_list))         # 2286
print(len(brown_list))          # 1525
print(len(blue_list))           # 1426
print(len(lightgreen_list))     # 1224
print(len(green_list))          # 993
print(len(red_list))            # 691
print(len(gray_list))           # 196
print(len(purple_list))         # 129
print(len(cyan_list))           # 109
print(len(black_list))          # 71
print(len(violet_list))         # 63
print(len(navy_list))           # 42
print(len(transparent_list))    # 25