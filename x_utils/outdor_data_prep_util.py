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
    h5_chunk_row_step_1M = 50*1000
    h5_chunk_row_step_10M = h5_chunk_row_step_1M * 10
    h5_chunk_row_step_100M = h5_chunk_row_step_1M * 100
    h5_chunk_row_step_1G = h5_chunk_row_step_1M * 1000
    h5_chunk_row_step =  h5_chunk_row_step_10M

    def __init__(self):
        print('Init Class OUTDOOR_DATA_PREP')
        #print(self.ETH_training_partAh5_folder)




    def add_geometric_scope_file(self,h5_file_name,line_num_limit=None):
        ''' calculate the geometric scope of raw h5 data, and add the result to attrs of dset'''
        begin = time.time()
        h5f = h5py.File(h5_file_name,'a')
        max_xyz = -np.ones((3))*1e10
        min_xyz = np.ones((3))*1e10

        xyz_dset = h5f['xyz']
        row_step = self.h5_chunk_row_step_1M
        print('There are %d lines in xyz dataset of file: %s'%(xyz_dset.shape[0],h5_file_name))
        #print('read row step = %d'%(row_step))

        for k in range(0,xyz_dset.shape[0],row_step):
            end = min(k+row_step,xyz_dset.shape[0])
            xyz_buf = xyz_dset[k:end,:]
            xyz_buf_max = xyz_buf.max(axis=0)
            xyz_buf_min = xyz_buf.min(axis=0)
            max_xyz = np.maximum(max_xyz,xyz_buf_max)
            min_xyz = np.minimum(min_xyz,xyz_buf_min)

            #if k/row_step % 100  == 0:
                #print('\nprocessing line %d , %0.3f in %s'%(k, float(k)/xyz_dset.shape[0] ,h5_file_name))
            if line_num_limit!=None and k > line_num_limit:
                print('break at k = ',line_num_limit)
                break
        max_str = '  '.join([ str(e) for e in max_xyz ])
        min_str = '  '.join([ str(e) for e in min_xyz ])
        xyz_dset.attrs['max'] = max_str
        xyz_dset.attrs['min'] = min_str
        print('File: %s\n\tmax_str=%s\n\tmin_str=%s'%(h5_file_name,max_str,min_str) )
        print('T=',time.time()-begin)


    def gen_rawETH_to_h5(self,label_files_glob,line_num_limit=None):
        '''
        transform the data and label to h5 format
        put every dim to a single dataset
            to speed up search and compare of a single dim
        data is large, chunk to speed up slice
        '''
        h5_chunk_row_step_1M = 50*1000
        h5_chunk_row_step_10M = h5_chunk_row_step_1M * 10
        h5_chunk_row_step_100M = h5_chunk_row_step_1M * 100
        h5_chunk_row_step_1G = h5_chunk_row_step_1M * 1000
        h5_chunk_row_step =  h5_chunk_row_step_10M
        h5_default_rows = h5_chunk_row_step * 5
        #compress = 1
        #print('compression_opts = ',compress)

        label_files_list = glob.glob(label_files_glob)
        data_files_list, h5_files_list = self.clean_label_files_list(label_files_list)
        print('%d data-label files detected'%(len(label_files_list)))
        for lf in label_files_list:
            print('\t%s'%(lf))

        for i,label_fn in enumerate(label_files_list):
            data_fn = data_files_list[i]
            h5_fn = h5_files_list[i]
            with open(data_fn,'r') as data_f, open(label_fn,'r') as label_f, h5py.File(h5_fn,'w') as h5_f:
                data_label_fs = itertools.izip(data_f,label_f)

                h5_f.attrs['file info'] = 'All the datasets are raw data downloaded from the ETH web'
                if 'xyz' in h5_f:
                    xyz_dset = h5_f['xyz']
                else:
                    xyz_dset = h5_f.create_dataset('xyz',shape=(h5_default_rows,3),maxshape=(None,3),dtype=np.float32,chunks=(h5_chunk_row_step,3))
                    #xyz_dset = h5_f.create_dataset('xyz',shape=(h5_default_rows,3),maxshape=(None,3),dtype=np.float32,chunks=(h5_chunk_row_step,3), compression='gzip',compression_opts=compress)
                if 'intensity' in h5_f:
                    intensity_dset = h5_f['intensity']
                else:
                    intensity_dset = h5_f.create_dataset('intensity',shape=(h5_default_rows,1),maxshape=(None,1),dtype=np.int32,chunks=(h5_chunk_row_step,1))
                    #intensity_dset = h5_f.create_dataset('intensity',shape=(h5_default_rows,1),maxshape=(None,1),dtype=np.int32,chunks=(h5_chunk_row_step,1), compression='gzip',compression_opts=compress)
                if 'color' in h5_f:
                    color_dset = h5_f['color']
                else:
                    color_dset = h5_f.create_dataset('color',shape=(h5_default_rows,3),maxshape=(None,3),dtype=np.uint8,chunks=(h5_chunk_row_step,3))
                    #color_dset = h5_f.create_dataset('color',shape=(h5_default_rows,3),maxshape=(None,3),dtype=np.uint8,chunks=(h5_chunk_row_step,3), compression='gzip',compression_opts=compress)
                if 'label' in h5_f:
                    label_dset = h5_f['label']
                else:
                    label_dset = h5_f.create_dataset('label',shape=(h5_default_rows,1),maxshape=(None,1),dtype=np.uint8,chunks=(h5_chunk_row_step,1))
                    #label_dset = h5_f.create_dataset('label',shape=(h5_default_rows,1),maxshape=(None,1),dtype=np.uint8,chunks=(h5_chunk_row_step,1), compression='gzip',compression_opts=compress)

                buf_rows = h5_chunk_row_step
                data_buf = np.zeros((buf_rows,7),np.float32)
                label_buf = np.zeros((buf_rows,1),np.int8)
                for k,data_label_line in enumerate(data_label_fs):
                    k_buf = k%buf_rows
                    data_buf[k_buf,:] =np.fromstring( data_label_line[0].strip(),dtype=np.float32,sep=' ' )
                    label_buf[k_buf,:] = np.fromstring( data_label_line[1].strip(),dtype=np.float32,sep=' ' )
                    if k_buf == buf_rows-1:
                        start = int(k/buf_rows)*buf_rows
                        end = k+1
                        print('start = %d, end = %d in file: %s'%(start,end,data_fn))
                        self.add_buf(xyz_dset,data_buf[:,0:3],start,end)
                        self.add_buf(intensity_dset,data_buf[:,3:4],start,end)
                        self.add_buf(color_dset,data_buf[:,4:7],start,end)
                        self.add_buf(label_dset,label_buf[:,0:1],start,end)
                        h5_f.flush()

                    if line_num_limit != None and k+1 >= line_num_limit:
                        print('break at k= ',k)
                        break

                self.add_buf_all(h5_f,xyz_dset,intensity_dset,color_dset,label_dset,data_buf,label_buf,k,buf_rows)
                self.cut_redundance(xyz_dset,k+1)
                self.cut_redundance(intensity_dset,k+1)
                self.cut_redundance(color_dset,k+1)
                self.cut_redundance(label_dset,k+1)

                print('having read %d lines from %s \n'%(k+1,data_fn))
                #print('h5 file line num = %d'%(xyz_dset.shape[0]))

    def cut_redundance(self,dset,file_rows):
        if dset.shape[0] > file_rows:
            dset.resize((file_rows,dset.shape[1]))
    def add_buf(self,dset,new_data,start,end):
        if dset.shape[0] < end:
            dset.resize((end,dset.shape[1]))
        dset[start:end,:] = new_data

    def add_buf_all(self,h5_f,xyz_dset,intensity_dset,color_dset,label_dset,data_buf,label_buf,k,buf_rows):
        k_buf = k%buf_rows
        start = int(k/buf_rows)*buf_rows
        end = k+1
        #print( 'start = %d, end = %d'%(start,end))
        self.add_buf(xyz_dset,data_buf[0:k_buf+1,0:3],start,end)
        self.add_buf(intensity_dset,data_buf[0:k_buf+1,3:4],start,end)
        self.add_buf(color_dset,data_buf[0:k_buf+1,4:7],start,end)
        self.add_buf(label_dset,label_buf[0:k_buf+1,0:1],start,end)
        h5_f.flush()
        #print('flushing k = ',k)

    def clean_label_files_list(self,label_files_list):
        data_files_list = []
        h5_files_list = []
        for i,label_file_name in enumerate(label_files_list):
            no_format_name = os.path.splitext(label_file_name)[0]
            data_file_name = no_format_name + '.txt'
            h5_file_name = no_format_name + '.hdf5'
            if not os.path.exists(data_file_name):
                label_files_list.pop(i)
                print('del label_files_list[%d]:%s'%(i,label_file_name))
            else:
                data_files_list.append(data_file_name)
                h5_files_list.append(h5_file_name)
        return data_files_list, h5_files_list



    #-------------------------------------------------------------------------------
    '''
    Implement Functions
    '''
    #-------------------------------------------------------------------------------

    def DO_add_geometric_scope_file(self):
        files_glob = os.path.join(self.ETH_training_partAh5_folder,'*.hdf5')
        #files_glob = os.path.join(self.ETH_training_partAh5_folder,'bildstein_station5_xyz_intensity_rgb.hdf5')
        files_list = glob.glob(files_glob)
        print('%d files detected'%(len(files_list)))

        IsMultiProcess = False
        line_num_limit = 1000*100
        line_num_limit = None

        if not IsMultiProcess:
            for file_name in files_list:
                self.add_geometric_scope_file(file_name,line_num_limit)
        else:
            mp_n = min(len(files_list),mp.cpu_count())
            pool = mp.Pool(mp_n)
            pool.imap_unordered(self.add_geometric_scope_file,files_list)


    def DO_gen_rawETH_to_h5(self,ETH_raw_labels_glob=None):
        if ETH_raw_labels_glob == None:
            labels_folder = '/home/x/Research/Dataset/ETH_Semantic3D_Dataset/training/part_A'
            labels_folder = '/short/dh01/yx2146/Dataset/ETH_Semantic3D_Dataset/training/part_B'
            #labels_folder = '/other/ETH_Semantic3D_Dataset/training/part_A
            ETH_raw_labels_glob = os.path.join(labels_folder,'*.labels')
        line_num_limit = None
        self.gen_rawETH_to_h5(ETH_raw_labels_glob)

#-------------------------------------------------------------------------------
'''
main
'''
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    outdoor_prep = OUTDOOR_DATA_PREP()
    outdoor_prep.DO_add_geometric_scope_file()
    #outdoor_prep.DO_gen_rawETH_to_h5()
    T = time.time() - START_T
    print('exit main, T = ',T)
