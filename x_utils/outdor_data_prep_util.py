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
    Local_training_partAh5_folder = '/home/x/Research/Dataset/ETH_Semantic3D_Dataset/training/part_A_h5'
    ETH_training_partAh5_folder = '/short/dh01/yx2146/Dataset/ETH_Semantic3D_Dataset/training/part_A_rawh5'
    ETH_training_partBh5_folder = '/short/dh01/yx2146/Dataset/ETH_Semantic3D_Dataset/training/part_B_rawh5'
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
        xyz_dset.attrs['max'] = max_xyz
        xyz_dset.attrs['min'] = min_xyz
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


    def get_blocked_dset(self,block_k):
        dset_name = str(block_k)
        #if self.block_dsets[block_k]!=None:
        if dset_name in self.h5f_blocked:
            #return self.block_dsets[block_k]
            return self.h5f_blocked[dset_name]
        rows_default = self.h5_chunk_row_step_100M
        dset = self.h5f_blocked.create_dataset( dset_name,shape=(rows_default,9),\
                maxshape=(None,9),dtype=np.float32,chunks=(self.h5_chunk_row_step,9) )
        dset.attrs['valid_num']=0
        block_scope_k = np.zeros((2,3))
        i_xyz = self.get_i_xyz(block_k)
        block_k = int( i_xyz[0]*self.block_nums[1]*self.block_nums[2] + i_xyz[1]*self.block_nums[2] + i_xyz[2] )
        block_scope_k[0,:] = i_xyz * self.block_step + self.xyz_min
        block_scope_k[1,:] = (i_xyz+1) * self.block_step + self.xyz_min
        dset.attrs['i_xyz'] = i_xyz
        dset.attrs['xyz_scope'] = block_scope_k
        #self.block_dsets[block_k]=dset
        self.h5f_blocked.flush()
        return dset

    def get_i_xyz(self,block_k):
        i_xyz = np.zeros(3,np.int64)
        i_xyz[2] = block_k % self.block_nums[2]
        k = int( block_k / self.block_nums[2] )
        i_xyz[1] = k % self.block_nums[1]
        k = int( k / self.block_nums[1] )
        i_xyz[0] = k % self.block_nums[0]
        return i_xyz

    def get_block_k(self,i_xyz):
        i_xyz = i_xyz.astype(np.uint64)
        block_k = int( i_xyz[0]*self.block_nums[1]*self.block_nums[2] + i_xyz[1]*self.block_nums[2] + i_xyz[2] )
        return block_k

    def get_block_index(self,xyz_k):
        i_xyz = ( (xyz_k - self.xyz_min)/self.block_step ).astype(np.int64)
        block_k = self.get_block_k(i_xyz)
#        i_xyz_test = self.get_i_xyz(block_k)
#        if (i_xyz_test != i_xyz).any():
#            print('get i_xyz ERROR!')
        return block_k


    def sort_to_blocks(self,file_name):
        '''split th ewhole scene to space sorted small blocks
        The whole scene is a group. Each block is one dataset in the group.
        The block attrs represents the field.
        '''
        self.block_step = np.ones((3))*20
#        IsMultiProcess = False
        self.row_num_limit = 1000*1000

        blocked_file_name = os.path.splitext(file_name)[0]+'_blocked.h5'
        with h5py.File(blocked_file_name,'w') as self.h5f_blocked,  h5py.File(file_name,'r') as h5_f:
            dset_list = [n for n in h5_f]
            #for dset_name in dset_list:
            xyz_dset = h5_f['xyz']
            label_dset = h5_f['label']
            color_dset = h5_f['color']
            intensity_dset = h5_f['intensity']
            self.h5f_blocked.attrs['elements'] = 'xyz-color-label-intensity-raw_i'
            self.total_row_num = xyz_dset.shape[0]
            self.xyz_max = xyz_dset.attrs['max']
            self.xyz_min = xyz_dset.attrs['min']
            xyz_scope = self.xyz_max - self.xyz_min

            self.block_nums = np.ceil(xyz_scope / self.block_step).astype(np.int32)
            blocks_N = self.block_nums[0] * self.block_nums[1] * self.block_nums[2]
            self.h5f_blocked.attrs['xyz_max'] = self.xyz_max
            self.h5f_blocked.attrs['xyz_min'] = self.xyz_min
            self.h5f_blocked.attrs['xyz_scope'] = xyz_scope
            self.h5f_blocked.attrs['block_step'] = self.block_step
            self.h5f_blocked.attrs['max_blocks_N'] = blocks_N
            print('blocks_N = ',blocks_N)

            row_step = self.h5_chunk_row_step_1M*8
            sorted_buf_dic = {}
##            if IsMultiProcess:
##                num_workers = min(mp.cpu_count(),6)
##                self.num_workers = num_workers
##                manager = mp.Manager()
##                row_queue = mp.Queue(num_workers)
##                sorted_buf_dic_mg = manager.dict()


            raw_buf = np.zeros((row_step,9))


            for k in range(0,xyz_dset.shape[0],row_step):
                end = min(k+row_step,xyz_dset.shape[0])
                t0_k = time.time()
                #print('start read %d:%d'%(k,end))
                raw_buf[:,0:3] = xyz_dset[k:end,:]
                raw_buf[:,3:6] = color_dset[k:end,:]
                raw_buf[:,6:7] = label_dset[k:end,:]
                raw_buf[:,7:8] = intensity_dset[k:end,:]
                raw_buf[:,8] = np.arange(raw_buf.shape[0]) + k
                t1_k = time.time()
                #print('all read T=',time.time()-read_t0)

                t2_0_k = time.time()

##                if not IsMultiProcess:
                sorted_buf_dic={}
                self.sort_buf(raw_buf,k,sorted_buf_dic)
##                else:
##                    sorted_buf_dic_mg = manager.dict()
##                    pool = []
##                    for i in range(num_workers):
##                        p = mp.Process(target=self.sort_onerow_inqueue_multi,args=(row_queue,sorted_buf_dic_mg,))
##                        p.start()
##                        pool.append(p)
##                    for j in range(row_step):
##                        row_queue.put(raw_buf[j,:])
##                        print('put ',j)
##
##                    for j in range(self.num_workers):
##                        row_queue.put(None)
##                    print('put all')
##                    for p in pool:
##                        p.join()
##                    sorted_buf_dic = sorted_buf_dic_mg._getvalue()

                t2_1_k = time.time()
                self.h5_write_buf(sorted_buf_dic)

                t2_2_k = time.time()
                self.h5f_blocked.flush()
                if int(k/row_step) % 1 == 0:
                     print('line: [%d,%d] blocked   block_T=%f s, read_T=%f ms, cal_t = %f ms, write_t= %f ms'%\
                           (k,end,time.time()-t0_k,(t1_k-t0_k)*1000,(t2_1_k-t2_0_k)*1000, (t2_2_k-t2_1_k)*1000 ))
                if hasattr(self,'row_num_limit') and self.row_num_limit!=None and  k+row_step>=self.row_num_limit:
                    print('break read at k= ',k)
                    break


            for i in range(blocks_N):
                if str(i) in self.h5f_blocked:
                #if self.block_dsets[i] != None:
                    dset_i = self.h5f_blocked[str(i)]
                    valid_n = dset_i.attrs['valid_num']
                    if dset_i.shape[0] > valid_n:
                        #print('original shape = ',self.block_dsets[i].shape)
                        dset_i.resize( (valid_n,dset_i.shape[1]) )
                        #print('resized to ',self.block_dsets[i].shape)

##    def sort_onerow_inqueue_multi(self,row_queue,sorted_buf_dic):
##        while True:
##            row = row_queue.get()
##            if row is None:
##                sorted_buf_dic.clear()
##                print('Meet the None row, in id: %d'%(os.getpid()))
##                return
##            #print(row)
##            self.sort_one_row(row,sorted_buf_dic)
##
##    def sort_one_row(self,raw_buf_i,sorted_buf_dic):
##        block_k = self.get_block_index(raw_buf_i[0:3])
##        dset_k =  self.get_blocked_dset(block_k)
##        row = raw_buf_i.reshape(1,-1)
##        if not block_k in sorted_buf_dic:
##            buf_k = []
##        else:
##            #sorted_buf_dic[block_k]=[]
##            buf_k = sorted_buf_dic[block_k]
##        buf_k.append(row)
##        dic_k = {block_k:buf_k}
##        sorted_buf_dic.update(dic_k)
##        #print(row)
##        #print(len(sorted_buf_dic[block_k]))

    def get_block_index_multi(self,raw_buf):
        block_ks = mp.Array('i',raw_buf.shape[0])
        num_workers = 2
        step = int(raw_buf.shape[0]/num_workers)
        pool = []
        for i in range(0,raw_buf.shape[0],step):
            end = min( (i+1)*step, raw_buf.shape[0])
            p = mp.Process(target=self.get_block_index_subbuf,args=(raw_buf[i:end,0:3],block_ks[i:end],) )
            p.start()
            pool.append(p)
        for p in pool:
            p.join()
        return block_ks


    def get_block_index_subbuf(self,sub_buf_xyz,sub_block_ks):
        for i in range(sub_buf_xyz.shape[0]):
            sub_block_ks[i] = self.get_block_index(sub_buf_xyz[i,0:3])

    def sort_buf(self,raw_buf,buf_start_k,sorted_buf_dic):
        #t0 = time.time()
        IsMulti = True
        if IsMulti:
            block_ks = self.get_block_index_multi(raw_buf)
        else:
            block_ks = np.zeros(raw_buf.shape[0],np.int64)
            for j in range(raw_buf.shape[0]):
                block_ks[j] = self.get_block_index(raw_buf[j,0:3])


        #t1 = time.time()
        for i in range(raw_buf.shape[0]):
            block_k = block_ks[i]
            #raw_buf[i,8] = block_k
            row = raw_buf[i,:].reshape(1,-1)
            #row[0,8] = i+buf_start_k
            if not block_k in sorted_buf_dic:
                sorted_buf_dic[block_k]=[]
            sorted_buf_dic[block_k].append(row)
        #t2 = time.time()
        #print('t1 = %d ms, t2 = %d ms'%( (t1-t0)*1000,(t2-t1)*1000 ))


    def h5_write_buf(self,sorted_buf_dic):
        for key in sorted_buf_dic:
            sorted_buf_dic[key] = np.concatenate(sorted_buf_dic[key],axis=0)
        for block_k in sorted_buf_dic:
            dset_k =  self.get_blocked_dset(block_k)
            valid_n = dset_k.attrs['valid_num']
            new_valid_n = valid_n + sorted_buf_dic[block_k].shape[0]
            if dset_k.shape[0] < new_valid_n:
                dset_k.resize(( dset_k.shape[0]+self.h5_chunk_row_step_10M,dset_k.shape[1]))
            dset_k[valid_n:new_valid_n,:] = sorted_buf_dic[block_k]
            dset_k.attrs['valid_num'] = new_valid_n


    #-------------------------------------------------------------------------------
    '''
    Implement Functions
    '''
    #-------------------------------------------------------------------------------

    def DO_add_geometric_scope_file(self):
        #files_glob = os.path.join(self.ETH_training_partBh5_folder,'*.hdf5')
        #files_glob = os.path.join(self.ETH_training_partAh5_folder,'*.hdf5')
        files_glob = os.path.join(self.Local_training_partAh5_folder,'*.hdf5')
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

    def Do_sort_to_blocks(self):
        partAh5_folder = '/short/dh01/yx2146/Dataset/ETH_Semantic3D_Dataset/training/part_A_rawh5'
        partAh5_folder = '/home/x/Research/Dataset/ETH_Semantic3D_Dataset/training/part_A_h5'
        ETH_raw_h5_glob =glob.glob(  os.path.join(partAh5_folder,'bildstein_station5_xyz_intensity_rgb.hdf5') )
        for fn in ETH_raw_h5_glob:
            self.sort_to_blocks(fn)

    def check_sorted_result(self):
        raw_file_name = '/short/dh01/yx2146/Dataset/ETH_Semantic3D_Dataset/training/part_A_rawh5/bildstein_station5_xyz_intensity_rgb.hdf5'
        sorted_file_name = '/short/dh01/yx2146/Dataset/ETH_Semantic3D_Dataset/training/part_A_rawh5/bildstein_station5_xyz_intensity_rgb_blocked.h5'
        with h5py.File(raw_file_name,'r') as raw_f, h5py.File(sorted_file_name,'r') as sorted_f:
            raw_xyz_set = raw_f['xyz']
            raw_color_set = raw_f['color']
            raw_label_set = raw_f['label']
            raw_intensity_set = raw_f['intensity']
            for block_k in sorted_f:
                dset_k = sorted_f[block_k]
                step = max(int(dset_k.shape[0]/20),1)
                for i in range(0,dset_k.shape[0],step):
                    sorted_d_i = dset_k[i,0:8]
                    raw_k = dset_k[i,-1]
                    raw_d_i = np.concatenate(  [raw_xyz_set[raw_k,:],raw_color_set[raw_k,:],raw_label_set[raw_k,:],raw_intensity_set[raw_k,:]] )
                    error = raw_d_i - sorted_d_i
                    err = np.absolute( error ).sum()
                    if err != 0:
                        print('sorted error: block_k=%s,i=%d'%(block_k,i))
                    #else:
                        #print('i=%d checked'%(i))
                self.check_sorted_dset_scope(dset_k)
            print('all checked')
    def check_sorted_dset_scope(self,dset):
        scope = dset.attrs['xyz_scope']
        xyz = dset[:,0:3]
        xyz_max = xyz.max(axis=0)
        xyz_min = xyz.min(axis=0)
        #print('\nscope criterion = \n',scope)
        #print('real min = \n',xyz_min,'\nreal max = \n',xyz_max)
        if (scope[1,:] > xyz_max).all() and (scope[0,:] < xyz_min).all():
            #print('scope checked OK')
            return True
        else:
            print('scope checked failed')
            return False

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def main():
    outdoor_prep = OUTDOOR_DATA_PREP()
    outdoor_prep.Do_sort_to_blocks()
    outdoor_prep.check_sorted_result()

    #outdoor_prep.DO_add_geometric_scope_file()
    #outdoor_prep.DO_gen_rawETH_to_h5()

if __name__ == '__main__':
    main()
    T = time.time() - START_T
    print('exit main, T = ',T)
