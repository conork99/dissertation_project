# virtual environment is used as miccai23
import os
import pandas as pd
import cv2
from PIL import Image
import h5py
import numpy as np

path = '//rfs02/rdm06/JointAI/King/FGADR-Seg-set_Release/Seg-set/'
input_path = path + 'Original_Images/'
SE_path = path + 'SoftExudate_Masks/'
EX_path = path + 'HardExudate_Masks/'
MA_path = path + 'Microaneurysms_Masks/'
HE_path = path + 'Hemohedge_Masks/'
save_path = 'M:/LIFE703/h5py_FGADR/h5py_FGADR/'

grade_label_path = path + 'DR_Seg_Grading_Label.csv'
df = pd.read_csv(grade_label_path, header=None)
name = df.loc[:, 0]
grade_label_all = df.loc[:, 1]
new_size = (128, 128)
for i in range(len(name)):
    img = Image.open(input_path + name[i]).resize(new_size, resample=Image.BILINEAR)
    SE_mask = Image.open(SE_path + name[i]).convert('L').resize(new_size, resample=Image.NEAREST)
    EX_mask = Image.open(EX_path + name[i]).convert('L').resize(new_size, resample=Image.NEAREST)
    MA_mask = Image.open(MA_path + name[i]).convert('L').resize(new_size, resample=Image.NEAREST)
    HE_mask = Image.open(HE_path + name[i]).convert('L').resize(new_size, resample=Image.NEAREST)
    grade_label = grade_label_all[i]

    save_h5py_path = path + '/' + 'h5py' + '/'
    h5f = h5py.File(save_path + '/' + name[i].replace('.png', '.h5'), 'a')
    h5f.create_dataset('img', data=img, compression='gzip', compression_opts=9)
    h5f.create_dataset('SE_mask', data=SE_mask, compression='gzip', compression_opts=9)
    h5f.create_dataset('EX_mask', data=EX_mask, compression='gzip', compression_opts=9)
    h5f.create_dataset('MA_mask', data=MA_mask, compression='gzip', compression_opts=9)
    h5f.create_dataset('HE_mask', data=HE_mask, compression='gzip', compression_opts=9)
    h5f.create_dataset('grade_label', data=grade_label)
    h5f.close()




