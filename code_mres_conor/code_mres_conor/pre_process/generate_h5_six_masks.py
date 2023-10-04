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
IRMA_path = path + 'IRMA_Masks/'
NV_path = path + 'Neovascularization_Masks/'
save_path = 'M:/LIFE703/h5py_FGADR/h5py_FGADR/'

grade_label_path = path + 'DR_Seg_Grading_Label.csv'
df = pd.read_csv(grade_label_path, header=None)
name = df.loc[:, 0]
grade_label_all = df.loc[:, 1]
new_size = (128, 128)
for i in range(len(name)):
    img = Image.open(input_path + name[i]).resize(new_size, resample=Image.BILINEAR)
    SE_mask = np.array(Image.open(SE_path + name[i]).convert('L').resize(new_size, resample=Image.NEAREST))
    if np.max(SE_mask) !=0:
        SE_mask = SE_mask / np.max(SE_mask)

    EX_mask = np.array(Image.open(EX_path + name[i]).convert('L').resize(new_size, resample=Image.NEAREST))
    if np.max(EX_mask) != 0:
        EX_mask = EX_mask / np.max(EX_mask)

    MA_mask = np.array(Image.open(MA_path + name[i]).convert('L').resize(new_size, resample=Image.NEAREST))
    if np.max(MA_mask) != 0:
        MA_mask = MA_mask / np.max(MA_mask)

    HE_mask = np.array(Image.open(HE_path + name[i]).convert('L').resize(new_size, resample=Image.NEAREST))
    if np.max(HE_mask) != 0:
        HE_mask = HE_mask / np.max(HE_mask)
    try:
        IRMA_mask = np.array(Image.open(IRMA_path + name[i]).convert('L').resize(new_size, resample=Image.NEAREST))
        if np.max(IRMA_mask) != 0:
            IRMA_mask = IRMA_mask / np.max(IRMA_mask)
    except:
        IRMA_mask = np.zeros((128, 128))
    try:
        NV_mask = np.array(Image.open(NV_path + name[i]).convert('L').resize(new_size, resample=Image.NEAREST))
        if np.max(NV_mask) != 0:
            NV_mask = NV_mask / np.max(NV_mask)
    except:
        NV_mask = np.zeros((128, 128))
    grade_label = grade_label_all[i]
    # print(np.max(SE_mask))
    # print(np.max(EX_mask))
    # print(np.max(MA_mask))
    # print(np.max(HE_mask))
    # print(np.max(IRMA_mask))
    # print(np.max(NV_mask))


    save_h5py_path = path + '/' + 'h5py' + '/'
    h5f = h5py.File(save_path + '/' + name[i].replace('.png', '.h5'), 'a')
    h5f.create_dataset('img', data=img, compression='gzip', compression_opts=9)
    h5f.create_dataset('SE_mask', data=SE_mask, compression='gzip', compression_opts=9)
    h5f.create_dataset('EX_mask', data=EX_mask, compression='gzip', compression_opts=9)
    h5f.create_dataset('MA_mask', data=MA_mask, compression='gzip', compression_opts=9)
    h5f.create_dataset('HE_mask', data=HE_mask, compression='gzip', compression_opts=9)
    h5f.create_dataset('IRMA_mask', data=IRMA_mask, compression='gzip', compression_opts=9)
    h5f.create_dataset('NV_mask', data=NV_mask, compression='gzip', compression_opts=9)
    h5f.create_dataset('grade_label', data=grade_label)
    h5f.close()




