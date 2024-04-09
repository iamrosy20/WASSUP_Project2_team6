import os
import pandas as pd
import shutil

# Read every file name
file_list = os.listdir('data/modified_images')

# Split file name and its extension
file_name = []
for one_file_list in file_list:
  name = one_file_list.split('.')[0]
  file_name.append(name)

# Load drug_data.csv
df = pd.read_csv('data/drug_data.csv')

# Check shape of pill and copy image into appropriate folder
for one_file in file_name:
  source = f'data/modified_images/{one_file}.png'

  index_num = df[df['품목일련번호'] == int(one_file)].index
  if df.iloc[index_num[0]]['의약품제형'] == '원형':        # circle 9908
    destination = f'data/shape/circle/{one_file}.png'
  elif df.iloc[index_num[0]]['의약품제형'] == '장방형':    # rectangle 7112
    destination = f'data/shape/rectangle/{one_file}.png'
  elif df.iloc[index_num[0]]['의약품제형'] == '타원형':    # ellipse 6712
    destination = f'data/shape/ellipse/{one_file}.png'
  elif df.iloc[index_num[0]]['의약품제형'] == '사각형':    # square 281
    destination = f'data/shape/square/{one_file}.png'
  elif df.iloc[index_num[0]]['의약품제형'] == '팔각형':    # octagon 275
    destination = f'data/shape/octagon/{one_file}.png'
  elif df.iloc[index_num[0]]['의약품제형'] == '삼각형':    # triangle 235
    destination = f'data/shape/triangle/{one_file}.png'
  elif df.iloc[index_num[0]]['의약품제형'] == '마름모형':  # rhombus 91
    destination = f'data/shape/rhombus/{one_file}.png'
  elif df.iloc[index_num[0]]['의약품제형'] == '오각형':    # pentagon 58
    destination = f'data/shape/pentagon/{one_file}.png'
  elif df.iloc[index_num[0]]['의약품제형'] == '육각형':    # hexagon 50
    destination = f'data/shape/hexagon/{one_file}.png'
  elif df.iloc[index_num[0]]['의약품제형'] == '반원형':    # semicircle 3
    destination = f'data/shape/semicircle/{one_file}.png'
  else:                                                   # etc 490
    destination = f'data/shape/etc/{one_file}.png'

  shutil.copyfile(source, destination)

circle_list = os.listdir('data/shape/circle')
rectangle_list = os.listdir('data/shape/rectangle')
ellipse_list = os.listdir('data/shape/ellipse')
square_list = os.listdir('data/shape/square')
octagon_list = os.listdir('data/shape/octagon')
triangle_list = os.listdir('data/shape/triangle')
rhombus_list = os.listdir('data/shape/rhombus')
pentagon_list = os.listdir('data/shape/pentagon')
hexagon_list = os.listdir('data/shape/hexagon')
semicircle_list = os.listdir('data/shape/semicircle')
etc_list = os.listdir('data/shape/etc')

print(len(circle_list))         # 9706
print(len(rectangle_list))      # 6948
print(len(ellipse_list))        # 6666
print(len(square_list))         # 276
print(len(octagon_list))        # 274
print(len(triangle_list))       # 235
print(len(rhombus_list))        # 91
print(len(pentagon_list))       # 58
print(len(hexagon_list))        # 50
print(len(semicircle_list))     # 3
print(len(etc_list))            # 479