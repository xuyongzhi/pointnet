#xyz

from __future__ import print_function
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import numpy as np
import h5py
import glob
import time
import multiprocessing as mp
import itertools
import argparse


START_T = time.time()

class OUTDOOR_DATA_PREP():
    ETH_training_partAh5_folder = '/short/dh01/yx2146/Dataset/ETH_Semantic3D_Dataset/training/part_A_h5'


    def __init__(self):
        print(self.ETH_training_partAh5_folder)

    def add_geometric_scope_file(self,h5_file_name,line_num_limit=None):
        ''' calculate the geometric scope of raw h5 data, and add the result to attrs of dset'''
        h5f = h5py.File(h5_file_name,'a')
        max_xyz = -np.ones((3))*1e10
        min_xyz = np.ones((3))*1e10

        xyz_dset = h5f['xyz']
        print('There are %d lines in xyz dataset of file: %s'%(xyz_dset.shape[0],h5_file_name))
        for k in range(xyz_dset.shape[0]):
            for i  in range(3):
                max_xyz[i] = max(max_xyz[i],xyz_dset[k,i])
                min_xyz[i] = min(min_xyz[i],xyz_dset[k,i])
            if k % int(xyz_dset.shape[0]/10) == 0:
                print('\nprocessing line %d - %0.3f in %s'%(k, k/xyz_dset.shape[0] ,h5_file_name))
                print('T = ',time.time()-START_T)
            if line_num_limit!=None and k > line_num_limit:
                print('break at k = ',line_num_limit)
                break
        max_str = '  '.join([ str(e) for e in max_xyz ])
        min_str = '  '.join([ str(e) for e in min_xyz ])
        xyz_dset.attrs['max'] = max_str
        xyz_dset.attrs['min'] = min_str
        print('File: %s\n\tmax_str=%s\n\tmin_str=%s'%(h5_file_name,max_str,min_str) )



    def DO_add_geometric_scope_file(self):
        files_glob = os.path.join(self.ETH_training_partAh5_folder,'*.hdf5')
        files_glob = os.path.join(self.ETH_training_partAh5_folder,'bildstein_station5_xyz_intensity_rgb.hdf5')
        files_list = glob.glob(files_glob)
        print('%d files detected'%(len(files_list)))

        IsMultiProcess = False
        line_num_limit = 10000

        if not IsMultiProcess:
            for file_name in files_list:
                self.add_geometric_scope_file(file_name,line_num_limit)
        else:
            mp_n = min(len(files_list),mp.cpu_count())
            pool = mp.Pool(mp_n)
            pool.imap_unordered(self.add_geometric_scope_file,files_list)


if __name__ == '__main__':
    outdoor_prep = OUTDOOR_DATA_PREP()
    outdoor_prep.DO_add_geometric_scope_file()
    T = time.time() - START_T
    print('exit main, T = ',T)
