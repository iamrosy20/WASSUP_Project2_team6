import os
import pandas as pd
import shutil

# Read every file name
file_list = os.listdir('data/modified_images/RL')

# Split file name and its extension
file_name = []
for one_file_list in file_list:
  if 'right' in one_file_list:
    if '.png' in one_file_list:
      name = one_file_list.split('.')[0]
      file_name.append(name[6:])

# Load drug_data.csv
df = pd.read_csv('data/drug_data.csv')

# Replace - value in front color with yellow color
df.replace({'색상앞': '-'}, '노랑', inplace=True)

# Replace - value in back color with front color
for index, row in df.iterrows():
  if (row['색상뒤'] == '-'):
    df.loc[index, '색상뒤'] = df.loc[index, '색상앞']

# Slice front color value based on the dividing line |
df['색상뒤'] = df['색상뒤'].str.slice(start=0, stop=2)

# Check color of pill and copy image into appropriate folder
for one_file in file_name:
  source = f'data/modified_images/RL/right_{one_file}.png'

  index_num = df[df['품목일련번호'] == int(one_file)].index
  if df.iloc[index_num[0]]['색상뒤'] == '하양':         # white 
    destination = f'data/color_back/white/{one_file}.png'
  elif df.iloc[index_num[0]]['색상뒤'] == '분홍':       # pink 
    destination = f'data/color_back/pink/{one_file}.png'
  elif df.iloc[index_num[0]]['색상뒤'] == '노랑':       # yellow 
    destination = f'data/color_back/yellow/{one_file}.png'
  elif df.iloc[index_num[0]]['색상뒤'] == '주황':       # orange 
    destination = f'data/color_back/orange/{one_file}.png'
  elif df.iloc[index_num[0]]['색상뒤'] == '갈색':       # brown 
    destination = f'data/color_back/brown/{one_file}.png'
  elif df.iloc[index_num[0]]['색상뒤'] == '파랑':       # blue 
    destination = f'data/color_back/blue/{one_file}.png'
  elif df.iloc[index_num[0]]['색상뒤'] == '연두':       # lightgreen 
    destination = f'data/color_back/lightgreen/{one_file}.png'
  elif df.iloc[index_num[0]]['색상뒤'] == '초록':       # green 
    destination = f'data/color_back/green/{one_file}.png'
  elif df.iloc[index_num[0]]['색상뒤'] == '빨강':       # red 
    destination = f'data/color_back/red/{one_file}.png'
  elif df.iloc[index_num[0]]['색상뒤'] == '회색':       # gray 
    destination = f'data/color_back/gray/{one_file}.png'
  elif df.iloc[index_num[0]]['색상뒤'] == '보라':       # purple 
    destination = f'data/color_back/purple/{one_file}.png'
  elif df.iloc[index_num[0]]['색상뒤'] == '청록':       # cyan 
    destination = f'data/color_back/cyan/{one_file}.png'
  elif df.iloc[index_num[0]]['색상뒤'] == '검정':       # black 
    destination = f'data/color_back/black/{one_file}.png'
  elif df.iloc[index_num[0]]['색상뒤'] == '자주':       # violet 
    destination = f'data/color_back/violet/{one_file}.png'
  elif df.iloc[index_num[0]]['색상뒤'] == '남색':       # navy 
    destination = f'data/color_back/navy/{one_file}.png'
  else:                                                 # transparent 
    destination = f'data/color_back/transparent/{one_file}.png'

  shutil.copyfile(source, destination)

white_list = os.listdir('data/color_back/white')
pink_list = os.listdir('data/color_back/pink')
yellow_list = os.listdir('data/color_back/yellow')
orange_list = os.listdir('data/color_back/orange')
brown_list = os.listdir('data/color_back/brown')
blue_list = os.listdir('data/color_back/blue')
lightgreen_list = os.listdir('data/color_back/lightgreen')
green_list = os.listdir('data/color_back/green')
red_list = os.listdir('data/color_back/red')
gray_list = os.listdir('data/color_back/gray')
purple_list = os.listdir('data/color_back/purple')
cyan_list = os.listdir('data/color_back/cyan')
black_list = os.listdir('data/color_back/black')
violet_list = os.listdir('data/color_back/violet')
navy_list = os.listdir('data/color_back/navy')
transparent_list = os.listdir('data/color_back/transparent')

print(len(white_list))          # 10288
print(len(pink_list))           # 3528
print(len(yellow_list))         # 3429
print(len(orange_list))         # 2166
print(len(brown_list))          # 1332
print(len(blue_list))           # 1058
print(len(lightgreen_list))     # 1205
print(len(green_list))          # 696
print(len(red_list))            # 553
print(len(gray_list))           # 219
print(len(purple_list))         # 62
print(len(cyan_list))           # 87
print(len(black_list))          # 65
print(len(violet_list))         # 48
print(len(navy_list))           # 5
print(len(transparent_list))    # 45