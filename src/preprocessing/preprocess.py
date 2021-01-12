from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
from unrar import rarfile


# cross-platform forward slash path notation
assets_folder = Path('./src/data')
temp_folder = Path('./src/preprocessing/temp')

# on linux, do pip install unrar and apt-get install libunrar5
file = rarfile.RarFile(str(assets_folder / 'data.rar'))
file.extractall(path=str(temp_folder))

# list absolute paths to every extracted file
original_rar_file_list = [os.path.join(r, file) for r,d,f in os.walk(temp_folder) for file in f]

# get only the rows/colmns containing data from CLIN_DIA.xlsx
images_diag_df = pd.read_excel(assets_folder / 'CLIN_DIA.xlsx')
images_diag_df.dropna(axis=0, how='all', inplace=True)
images_diag_df.dropna(axis=1, how='all', inplace=True)

# filter out non-existent images and files where diagnose wasn't finshed yet
def construct_image_path(image_diag_row):
  """Return the path for the image in the diag df, if it exists, otherwise, return NaN"""
  try:
    return [path for path in original_rar_file_list if image_diag_row['id'].upper()+'.BMP' in path][0]
  except:
    return np.nan

images_diag_df['filepath'] = images_diag_df.apply(construct_image_path, axis = 1)
images_diag_df.dropna(axis = 0, subset=['filepath'], inplace=True)
images_diag_df = images_diag_df[images_diag_df['kat.Diagnose'].isin([1,2,3])]

# create binary label based on kat.Diagnose
# 1		==> no_doctor
# 2/3	==> doctor
images_diag_df['label'] = images_diag_df['kat.Diagnose'].apply(lambda x: 'no_doctor' if x == 1 else 'doctor')

# split the images_diag_df in train and test with each 22% cases that should see a doctor
images_diag_train, images_diag_test = train_test_split(images_diag_df, test_size=0.1, random_state=101, stratify=images_diag_df['label'])
print('% see doctor cases - train:', images_diag_train[images_diag_train['label'] == 'doctor'].shape[0]/len(images_diag_train))
print('% see doctor cases - test:', images_diag_test[images_diag_test['label'] == 'doctor'].shape[0]/len(images_diag_test))


# create directory structure
structure = [ 'train', 'test', 'train/no_doctor', 'train/doctor', 'test/no_doctor', 'test/doctor']
for dir in structure:
  dir = temp_folder / dir
  if not dir.exists():
    os.mkdir(dir)

current_rows = 'train'
progress_counter = 0

for rows in (images_diag_train.iterrows(), images_diag_test.iterrows()):
  for index, row in rows:
    # pre-processing part
    image = cv2.imread(f"{row['filepath']}")
    # save to image tree
    if current_rows == 'train':
      if row['label'] == 'no_doctor':
        cv2.imwrite(str(temp_folder / 'train' / 'no_doctor' / f"{row['id']}.png"), image)
      else:
        cv2.imwrite(str(temp_folder / 'train' / 'doctor' / f"{row['id']}.png"), image)
    else:
      if row['label'] == 'no_doctor':
        cv2.imwrite(str(temp_folder / 'test' / 'no_doctor' / f"{row['id']}.png"), image)
      else:
        cv2.imwrite(str(temp_folder / 'test' / 'doctor' / f"{row['id']}.png"), image)
    progress_counter += 1
    if progress_counter % 200 == 0:
      print(f'{progress_counter/images_diag_df.shape[0]:.1%}')
  current_rows = 'test'

