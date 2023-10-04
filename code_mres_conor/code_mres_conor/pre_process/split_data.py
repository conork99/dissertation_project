import os

te_path = 'M:/LIFE703/h5py_FGADR/h5py_FGADR/'
c_path = 'M:/LIFE703/tvt_list/tvt_list/'
f_te = open(c_path + 'train.list', 'w')

for name in os.listdir(te_path):
    h5py_path = te_path + '/' + name
    f_te.write(name + '\n')