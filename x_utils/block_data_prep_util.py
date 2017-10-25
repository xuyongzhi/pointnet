#xyz

from __future__ import print_function
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
#from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import numpy as np
import h5py
import glob
import time
import multiprocessing as mp
import itertools
#import argparse


START_T = time.time()

g_h5_num_row_1M = 50*1000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
UPER_DIR = os.path.dirname(ROOT_DIR)

#DATASET_NAME = 'ETH'
DATASET_NAME = 'STANFORD_INDOOR3D'

class GLOBAL_PARA():
    stanford_indoor3d_collected_path = os.path.join(ROOT_DIR,'data/stanford_indoor3d')
    stanford_indoor3d_rawh5 = os.path.join(ROOT_DIR,'data/stanford_indoor3d_rawh5')
    stanford_indoor3d_sortedh5 = os.path.join(ROOT_DIR,'data/stanford_indoor3d_sortedh5')
    stanford_indoor3d_stride_0d5_step_0d5 = os.path.join(ROOT_DIR,'data/stanford_indoor3d_sortedh5_stride_0.5_step_0.5')
    stanford_indoor3d_stride_0d5_step_1 = os.path.join(ROOT_DIR,'data/stanford_indoor3d_sortedh5_stride_0.5_step_1')
    stanford_indoor3d_stride_0d5_step_1_4096 = os.path.join(ROOT_DIR,'data/stanford_indoor3d_sortedh5_stride_0.5_step_1_4096')
    stanford_indoor3d_globalnormedh5_stride_0d5_step_1_4096 = os.path.join(ROOT_DIR,'data/stanford_indoor3d_globalnormedh5_stride_0.5_step_1_4096')

    ETH_traing_A =  os.path.join(UPER_DIR,'Dataset/ETH_Semantic3D_Dataset/training')
    ETH_raw_partA = os.path.join( ETH_traing_A,'partA' )
    ETH_A_rawh5 = os.path.join( ETH_traing_A,'part_A_rawh5' )
    ETH_A_stride_1_step_1 = os.path.join( ETH_traing_A, 'A_stride_1_step_1' )
    ETH_A_stride_2_step_2 = os.path.join( ETH_traing_A, 'A_stride_2_step_2' )
    ETH_A_stride_4_step_4 = os.path.join( ETH_traing_A, 'A_stride_4_step_4' )
    ETH_A_stride_4_step_8 = os.path.join( ETH_traing_A, 'A_stride_4_step_8' )
    ETH_A_stride_5_step_5 = os.path.join( ETH_traing_A, 'A_stride_5_step_5' )
    ETH_A_stride_8_step_8 = os.path.join( ETH_traing_A, 'A_stride_8_step_8' )
    ETH_A_stride_20_step_10 = os.path.join( ETH_traing_A, 'A_stride_20_step_10' )

    seg_train_path = os.path.join(ROOT_DIR,'x_sem_seg/ETH3D_sem_seg_hdf5_data')

    h5_num_row_1M = 50*1000
    h5_num_row_10M = h5_num_row_1M * 10
    h5_num_row_100M = h5_num_row_1M * 100
    h5_num_row_1G = h5_num_row_1M * 1024
    h5_chunk_row_step =  h5_num_row_1M

    @classmethod
    def sample(cls,org_N,sample_N,sample_method='random'):
        if sample_method == 'random':
            if org_N == sample_N:
                sample_choice = np.arange(sample_N)
            elif org_N > sample_N:
                sample_choice = np.random.choice(org_N,sample_N)
                #reduced_num += org_N - sample_N
            else:
                #sample_choice = np.arange(org_N)
                new_samp = np.random.choice(org_N,sample_N-org_N)
                sample_choice = np.concatenate( (np.arange(org_N),new_samp) )
            #str = '%d -> %d  %d%%'%(org_N,sample_N,100.0*sample_N/org_N)
            #print(str)
        return sample_choice


def rm_file_name_midpart(fn,rm_part):
    base_name = os.path.basename(fn)
    parts = base_name.split(rm_part)
    if len(parts)>1:
        new_bn = parts[0] + parts[1]
    else:
        new_bn = parts[0]
    new_fn = os.path.join(os.path.dirname(fn),new_bn)
    return new_fn



class Raw_H5f():
    '''
    (1) raw:unsorted,all the time in one dataset
    (2) flexible  for different data types: each type in one dataset
    (3) class "Sorted_H5f" will sort data to blocks based on this class
    '''
    file_flag = 'RAW_H5F'
    h5_num_row_1M = 50*1000
    dtypes = { 'xyz':np.float32, 'intensity':np.int32, 'color':np.uint8,'label':np.uint8 }
    num_channels = {'xyz':3,'intensity':1,'color':3,'label':1}
    def __init__(self,raw_h5_f,file_name):
        self.raw_h5f = raw_h5_f
        self.get_summary_info()
        self.file_name = file_name
        self.num_default_row = 0

    def set_num_default_row(self,N):
        self.num_default_row = N

    def get_dataset(self,data_name):
        if data_name in self.raw_h5f:
            return self.raw_h5f[data_name]
        assert(data_name in self.dtypes)
        nc = self.num_channels[data_name]
        dset = self.raw_h5f.create_dataset(data_name,shape=(self.num_default_row,nc),\
                                    maxshape=(None,nc),dtype=self.dtypes[data_name],\
                                    chunks = (self.h5_num_row_1M,nc),\
                                    compression = "gzip")
        dset.attrs['valid_num'] = 0
        setattr(self,data_name+'_dset',dset)
        return dset
    def get_total_num_channels_name_list(self):
        total_num_channels = 0
        data_name_list = [str(dn) for dn in self.raw_h5f]
        for dn in data_name_list:
            total_num_channels += self.num_channels[dn]

        return total_num_channels,data_name_list

    def append_to_dset(self,dset_name,new_data):
       self.add_to_dset(dset_name,new_data,None,None)

    def get_all_dsets(self,start_idx,end_idx):
        out_dset_order = ['xyz','color','label','intensity']
        data_list = []
        for dset_name in out_dset_order:
            if dset_name in self.raw_h5f:
                data_k = self.raw_h5f[dset_name][start_idx:end_idx,:]
                data_list.append(data_k)
        data = np.concatenate(data_list,1)
        return data

    def add_to_dset(self,dset_name,new_data,start,end):
        dset = self.get_dataset(dset_name)
        valid_n  = dset.attrs['valid_num']
        if start == None:
            start = valid_n
            end = start + new_data.shape[0]
        if dset.shape[0] < end:
            dset.resize((end,dset.shape[1:]))
        if valid_n < end:
            dset.attrs['valid_num'] = end
        dset[start:end,:] = new_data

    def rm_invalid(self):
        for dset_name in self.raw_h5f:
            dset = self.raw_h5f[dset_name]
            if 'valid_num' in dset.attrs:
                valid_num = dset.attrs['valid_num']
                if valid_num < dset.shape[0]:
                    dset.resize( (valid_num,dset.shape[1:]) )

    def get_summary_info(self):
        for dset_name in self.raw_h5f:
            setattr(self,dset_name+'_dset',self.raw_h5f[dset_name])
        if 'xyz' in self.raw_h5f:
            self.total_row_N = self.xyz_dset.shape[0]
            self.xyz_max = self.xyz_dset.attrs['max']
            self.xyz_min = self.xyz_dset.attrs['min']
            self.xyz_scope = self.xyz_max - self.xyz_min

    def generate_objfile(self,obj_file_name,IsLabelColor):
        with open(obj_file_name,'w') as out_obj_file:
            xyz_dset = self.xyz_dset
            color_dset = self.color_dset
            label_dset = self.label_dset

            row_step = self.h5_num_row_1M * 10
            row_N = xyz_dset.shape[0]
            for k in range(0,row_N,row_step):
                end = min(k+row_step,row_N)
                xyz_buf_k = xyz_dset[k:end,:]
                color_buf_k = color_dset[k:end,:]
                buf_k = np.hstack((xyz_buf_k,color_buf_k))
                label_k = label_dset[k:end,0]
                for j in range(0,buf_k.shape[0]):
                    if not IsLabelColor:
                        str_j = 'v   ' + '\t'.join( ['%0.5f'%(d) for d in  buf_k[j,0:3]]) + '  \t'\
                        + '\t'.join( ['%d'%(d) for d in  buf_k[j,3:6]]) + '\n'
                    else:
                        label = label_k[j]
                        label_color = Normed_H5f.g_label2color[label]
                        str_j = 'v   ' + '\t'.join( ['%0.5f'%(d) for d in  buf_k[j,0:3]]) + '  \t'\
                        + '\t'.join( ['%d'%(d) for d in  label_color ]) + '\n'
                    out_obj_file.write(str_j)

                rate = int(100.0 * end / row_N)
                e = row_step / row_N
                if rate > 3 and rate % 3 <= e:
                    print('gen raw obj: %d%%'%(rate))
                if rate > 3:
                    break

    def create_done(self):
        self.rm_invalid()
        self.add_geometric_scope()

    def add_geometric_scope(self,line_num_limit=None):
        ''' calculate the geometric scope of raw h5 data, and add the result to attrs of dset'''
        #begin = time.time()
        max_xyz = -np.ones((3))*1e10
        min_xyz = np.ones((3))*1e10

        xyz_dset = self.xyz_dset
        row_step = self.h5_num_row_1M
        print('File: %s   %d lines'\
              %(os.path.basename(self.file_name),xyz_dset.shape[0]) )
        #print('read row step = %d'%(row_step))

        for k in range(0,xyz_dset.shape[0],row_step):
            end = min(k+row_step,xyz_dset.shape[0])
            xyz_buf = xyz_dset[k:end,:]
            xyz_buf_max = xyz_buf.max(axis=0)
            xyz_buf_min = xyz_buf.min(axis=0)
            max_xyz = np.maximum(max_xyz,xyz_buf_max)
            min_xyz = np.minimum(min_xyz,xyz_buf_min)

            if line_num_limit!=None and k > line_num_limit:
                print('break at k = ',line_num_limit)
                break
        xyz_dset.attrs['max'] = max_xyz
        xyz_dset.attrs['min'] = min_xyz
        max_str = '  '.join([ str(e) for e in max_xyz ])
        min_str = '  '.join([ str(e) for e in min_xyz ])
        print('max_str=%s\tmin_str=%s'%(max_str,min_str) )
        #print('T=',time.time()-begin)


class Sorted_H5f():
    '''
    (1) sorted: sort raw h5f by position to blocks, each block in one dataset
    (2) store all types of data (xyz,color,intensity,label..) together (float32)
    '''
    file_flag = 'SORTED_H5F'
    data_name_list_candidate = ['xyz','color','label','intensity','org_row_index']
    data_channels = {'xyz':3,'color':3,'label':1,'intensity':1,'org_row_index':1}
    IS_CHECK = False # when true, store org_row_index
    data_idxs = {}
    total_num_channels = 0

    actions = ''
    stride_to_align = 1
    h5_num_row_1M = g_h5_num_row_1M


    def normalize_dset(self,block_k_str,xyz_1norm_method='global'):
        '''
        (1) xyz/max
        (2) xy-min-block_size/2  (only xy)
        (3) color / 255
        '''
        raw_dset_k = self.sorted_h5f[block_k_str]

        norm_data_dic = {}
        raw_xyz = raw_dset_k[:,self.data_idxs['xyz']]
        #  xyz_1norm
        if xyz_1norm_method == 'global': # used by QI
            # 1norm within the whole scene
            # use by QI in indoor. Since room scale is not large, this is fine.
            # For outdoor,a scene could be too large, maybe not a good choice
            IsUseAligned = False
            if IsUseAligned:
                file_scene_zero = self.xyz_min_aligned
                file_scene_scope = self.xyz_max_aligned - self.xyz_min_aligned
            else:
                file_scene_zero = self.sorted_h5f.attrs['xyz_min']
                file_scene_scope = self.sorted_h5f.attrs['xyz_max'] - self.sorted_h5f.attrs['xyz_min']
            xyz_1norm = (raw_xyz - file_scene_zero) / file_scene_scope
        elif xyz_1norm_method == 'local':
            # 1norm within the block
            block_scope = raw_dset_k.attrs['xyz_max'] - raw_dset_k.attrs['xyz_min']
            xyz_1norm = (raw_xyz-raw_dset_k.attrs['xyz_min']) / block_scope

        # xyz_midnorm
        xyz_midnorm = raw_xyz+0 # as a new variable, not a reference
        # only norm x,y. Keep z be the raw value
        #xyz_min_real = np.min(raw_xyz,axis=0)
        #xyz_midnorm[:,0:2] -= (xyz_min_real[0:2] + self.block_step[0:2]/2)  # used by QI
        block_mid = (raw_dset_k.attrs['xyz_min'] + raw_dset_k.attrs['xyz_max'] ) / 2
        xyz_midnorm[:,0:2] -= block_mid[0:2]  # I think is better
        # for z, just be positive
        xyz_midnorm[:,2] -= self.sorted_h5f.attrs['xyz_min'][2]

        norm_data_dic['xyz_midnorm'] = xyz_midnorm
        norm_data_dic['xyz_1norm'] = xyz_1norm

        # color_1norm
        if 'color' in self.data_idxs:
            color_1norm = raw_dset_k[:,self.data_idxs['color']] / 255.0
            norm_data_dic['color_1norm']=color_1norm


        # intensity_1norm
        if 'intensity' in self.data_idxs:
            # ETH senmantic3D intensity range from -2047 to 2048
            intensity = raw_dset_k[:,self.data_idxs['intensity']]
            intensity_1norm = (intensity+2047)/(2048+2047)
            norm_data_dic['intensity_1norm']=intensity_1norm

        # order: 'xyz_midnorm' 'color_1norm' ''xyz_1norm' intensity_1norm'
        norm_data_list = []
        for data_name in Normed_H5f.data_elements:
            if data_name in norm_data_dic:
                norm_data_list.append(norm_data_dic[data_name])
        data_norm = np.concatenate( norm_data_list,1 )

        label = raw_dset_k[:,self.data_idxs['label'][0]]
        return data_norm,label,raw_xyz

    def __init__(self,sorted_h5f,file_name=None):
        self.sorted_h5f = sorted_h5f
        self.get_summary_info()
        if file_name != None:
            self.file_name = file_name
        else:
            self.file_name = None
        self.reduced_num = 0
        self.update_data_index()

    def get_summary_info(self):
        root_attrs = ['total_row_N','total_block_N',\
                      'block_step','block_stride','block_dims_N',\
                      'xyz_min_aligned','xyz_max_aligned','xyz_scope_aligned']
        for attr in root_attrs:
            if attr in self.sorted_h5f.attrs:
                setattr(self,attr,self.sorted_h5f.attrs[attr])
                #print(attr,getattr(self,attr) )

    def show_summary_info(self):
        print('\n\nsummary of file: ',self.file_name)
        root_attrs = ['total_row_N','total_block_N','block_step',\
                      'block_stride','block_dims_N','xyz_min_aligned','xyz_max_aligned',\
                      'xyz_scope_aligned']
        for attr in root_attrs:
            if attr in self.sorted_h5f.attrs:
                print(attr,getattr(self,attr) )
        for k, dset_n in enumerate(self.sorted_h5f):
            dset = self.sorted_h5f[dset_n]
            if k < 3:
                print('dset ',dset_n,'  shape=[%d,%d]'%(dset.shape[0],dset.shape[1]))
                for attr in dset.attrs:
                    print(attr,' = ',dset.attrs[attr])
                print(dset[0:min(4,dset.shape[0]),:])
                print(dset[-1,:])

    def update_data_index(self,data_name_list_In=None):
        if data_name_list_In == None:
            data_name_list_In = self.data_name_list_candidate[0:3]
        data_index = {}
        last_index = 0
        if self.IS_CHECK and 'org_row_index' not in data_name_list_In:
            data_name_list_In += ['org_row_index']
        data_name_list_In = set(data_name_list_In)
        for dn in self.data_name_list_candidate:
            if dn in data_name_list_In:
                data_index[dn] = range(last_index,last_index+self.data_channels[dn])
                last_index += self.data_channels[dn]
        self.data_idxs = data_index
        self.total_num_channels = last_index

    def set_root_attr(self,attr,value):
        setattr(self,attr,value)
        self.sorted_h5f.attrs[attr] = value

    def add_total_row_block_N(self):
        total_row_N = 0
       # if 'total_row_N' in self.sorted_h5f.attrs and 'total_block_N' in self.sorted_h5f.attrs:
       #     print('both row block _N exist, no need to cal')
       #     return self.total_row_N, self.total_block_N
        n = -1
        for n,dn in enumerate( self.sorted_h5f ):
            total_row_N += self.sorted_h5f[dn].shape[0]
            #if n % 200 == 0:
                #print('dset: ',dn, '   file_name= ',self.file_name,'  in add_total_row_block_N')
            #if n > 10:
                #break

        self.set_root_attr('total_row_N',total_row_N)
        self.set_root_attr('total_block_N',n+1)
        print('add_total_row_block_N:  file: %s \n   total_row_N = %d,  total_block_N = %d'%( os.path.basename(self.file_name),self.total_row_N,self.total_block_N))
        return total_row_N, n+1

    def copy_root_summaryinfo_from_another(self,h5f0,copy_flag):
        if copy_flag =='new_stride':
            attrs = ['xyz_max','xyz_min']
        elif copy_flag == 1:
            attrs = ['xyz_max','xyz_min','total_row_N','total_block_N']
        elif copy_flag == 'sub': # sampled
            attrs = ['xyz_max','xyz_min','block_step','block_stride']
        elif copy_flag == 'all':
            attrs = ['xyz_max','xyz_min','total_row_N','total_block_N','block_step','block_stride']
        else:
            attrs = ['xyz_max','xyz_min']

        for attr in attrs:
            if attr in h5f0.attrs:
                self.sorted_h5f.attrs[attr] = h5f0.attrs[attr]
        self.get_summary_info()
        if hasattr(self,'block_step') and hasattr(self,'block_stride'):
            self.set_step_stride(self.block_step,self.block_stride)

    def copy_root_geoinfo_from_raw(self,raw_h5f,attrs=['xyz_max','xyz_min']):
        for attr in attrs:
            if hasattr(raw_h5f,attr):
                self.sorted_h5f.attrs[attr] = getattr(raw_h5f,attr)
        _,data_name_list = raw_h5f.get_total_num_channels_name_list()
        self.update_data_index(data_name_list)
        self.get_summary_info()

    def set_step_stride(self,block_step,block_stride):
        xyz_min = self.sorted_h5f.attrs['xyz_min']
        xyz_max = self.sorted_h5f.attrs['xyz_max']
        xyz_min_aligned = xyz_min - xyz_min % self.stride_to_align - [0,0,1]
        xyz_max_aligned = xyz_max - xyz_max % 1 + 1
        xyz_scope_aligned =  xyz_max_aligned - xyz_min_aligned
        # step or stride ==-1 means one step/stride the whole scene
        for i in range(0,3):
            if block_step[i]  == -1:
                block_step[i] = xyz_scope_aligned[i]
            if block_stride[i]  == -1:
                block_stride[i] = xyz_scope_aligned[i]
        block_dims_N = np.ceil(xyz_scope_aligned / block_stride).astype(np.int64)
        self.sorted_h5f.attrs['block_step'] = block_step
        self.sorted_h5f.attrs['block_stride'] = block_stride
        self.sorted_h5f.attrs['block_dims_N'] = block_dims_N
        self.sorted_h5f.attrs['xyz_min_aligned'] = xyz_min_aligned
        self.sorted_h5f.attrs['xyz_max_aligned'] = xyz_max_aligned
        self.sorted_h5f.attrs['xyz_scope_aligned'] = xyz_scope_aligned
        self.get_summary_info()

    def block_index_to_ixyz(self,block_k):
        i_xyz = np.zeros(3,np.int64)
        i_xyz[2] = block_k % self.block_dims_N[2]
        k = int( block_k / self.block_dims_N[2] )
        i_xyz[1] = k % self.block_dims_N[1]
        k = int( k / self.block_dims_N[1] )
        i_xyz[0] = k % self.block_dims_N[0]
        return i_xyz

    def ixyz_to_block_index(self,i_xyz):
        i_xyz = i_xyz.astype(np.uint64)
        block_k = int( i_xyz[0]*self.block_dims_N[1]*self.block_dims_N[2] + i_xyz[1]*self.block_dims_N[2] + i_xyz[2] )
        return block_k
    def xyz_to_block_index(self,xyz_k):
        assert((self.block_step == self.block_stride).all()),"step != stride,the out k is not unique"

        #i_xyz = ( (xyz_k - self.raw_h5f.xyz_min)/block_step ).astype(np.int64)
        i_xyz = ( (xyz_k - self.xyz_min_aligned)/self.block_stride ).astype(np.int64)
        block_k = self.ixyz_to_block_index(i_xyz)
       # i_xyz_test = self.block_index_to_ixyz(block_k)
       # if (i_xyz_test != i_xyz).any():
       #     print('get i_xyz ERROR!')
        return block_k


    def get_sub_block_ks(self,block_k,new_sorted_h5f):
        '''
        For the space k in current file,
        return the corresponding block_ks in a new file with new step and stride
        block_ks is a list
        '''
        i_xyz = self.block_index_to_ixyz(block_k)
        i_xyz_new_start = i_xyz * self.block_stride / new_sorted_h5f.block_stride
        i_xyz_new_start = (i_xyz_new_start).astype(np.int)
        #print( self.xyz_min_aligned )
        #print( new_sorted_h5f.xyz_min_aligned )
        i_xyz_new_list = []
        block_k_new_list = []

        # for check
        IsCheck_Scope =  False
        if IsCheck_Scope:
            min_k,max_k,_ = self.get_block_scope_from_k(block_k)

        if (self.block_step > new_sorted_h5f.block_step).any():
            '''
            find all the small(out) blocks within the large input block
            The out dataset is a base dataset in which: block_step_out == block_stride_out
            '''
            assert((self.block_step > new_sorted_h5f.block_step ).all())
            assert((new_sorted_h5f.block_step == new_sorted_h5f.block_stride).all())

            search = np.ceil(self.block_step / new_sorted_h5f.block_step).astype(np.int64)
            for i_x in range(0,search[0]):
                for i_y in range(0,search[1]):
                    for i_z in range(0,search[2]):
                        i_xyz_new = ( i_xyz_new_start + np.array([i_x,i_y,i_z]) ).astype(np.uint64)
                        block_k_new = new_sorted_h5f.ixyz_to_block_index(i_xyz_new)

                        #check
                        if IsCheck_Scope:
                            min_k_new,max_k_new,_ = new_sorted_h5f.get_block_scope_from_k(block_k_new)
                            min_check = (min_k_new >= min_k).all()
                            max_check = (max_k_new <= max_k).all()
                        else:
                            min_check = True
                            max_check = True
                        if not min_check & max_check:
                            print('new=small failed i_xyz=',[i_x,i_y,i_z])
                            if not min_check:
                                print('\nmin check failed in get_sub_blcok_ks')
                                print('new min = ',min_k_new,'\norg min = ',min_k)
                            if not max_check:
                                print('\nmax check failed in get_sub_blcok_ks')
                                print('new max = ',max_k_new,'\norg max = ',max_k)

                        else:
                            i_xyz_new_list.append(i_xyz_new)
                            block_k_new_list.append(block_k_new)
                            #print('both min and max check passed')

        else:
            '''
            find all the large(out) blocks contains the small input block
            check: all xyz_scope_k_new contain xyz_scope_k
            '''
            assert( (self.block_step <= new_sorted_h5f.block_step).all() )
            assert( (new_sorted_h5f.block_step % self.block_step == 0).all() )
            assert( (new_sorted_h5f.block_stride >= self.block_step).all() )
            assert( (new_sorted_h5f.block_stride % self.block_step == 0).all() )


            search = ( new_sorted_h5f.block_step / new_sorted_h5f.block_stride ).astype(np.float64)
            if ( search%1*new_sorted_h5f.block_stride >= self.block_step).all() :
                search = np.ceil(search).astype(np.int64)
            else:
                search = np.trunc(search).astype(np.int64)
            for i_x in range( -search[0]+1,1 ):
                for i_y in range(  -search[1]+1,1  ):
                    for i_z in range(  -search[2]+1,1 ):
                        i_xyz_new = ( i_xyz_new_start + np.array([i_x,i_y,i_z]) ).astype(np.int64)
                        if ( (i_xyz_new < 0).any() or (i_xyz_new > new_sorted_h5f.block_dims_N).any() ):
                            continue

                        block_k_new = new_sorted_h5f.ixyz_to_block_index(i_xyz_new)
                        # check
                        if IsCheck_Scope:
                            min_k_new,max_k_new,_ = new_sorted_h5f.get_block_scope_from_k(block_k_new)
                            min_check = (min_k_new <= min_k).all()
                            max_check = (max_k_new >= max_k).all()
                        else:
                            min_check = True
                            max_check = True

                        if not min_check & max_check:
                            print('new=large failed i_xyz=',[i_x,i_y,i_z])
                            if not min_check:
                                print('\nmin check failed in get_sub_blcok_ks')
                                print('new min = ',min_k_new,'\norg min = ',min_k)
                            if not max_check:
                                print('\nmax check failed in get_sub_blcok_ks')
                                print('new max = ',max_k_new,'\norg max = ',max_k)

                        else:
                            #print('both min and max check passed, i_xyz= ',[i_x,i_y,i_z])
                            i_xyz_new_list.append(i_xyz_new)
                            block_k_new_list.append(block_k_new)
        return block_k_new_list,i_xyz_new_list

    def get_blocked_dset(self,block_k,new_set_default_rows=None,column_N = 9):
        if not type(block_k) is int:
            block_k = int(block_k)

        dset_name = str(block_k)
        if dset_name in self.sorted_h5f:
            return self.sorted_h5f[dset_name]
        if new_set_default_rows==None:
            new_set_default_rows = self.h5_num_row_1M
        #dset = self.h5f_blocked.create_dataset( dset_name,shape=(new_set_default_rows,n),\
                #maxshape=(None,n),dtype=np.float32,chunks=(self.h5_num_row_1M/5,n) )
        dset = self.sorted_h5f.create_dataset( dset_name,shape=(new_set_default_rows,column_N),\
                maxshape=(None,column_N),dtype=np.float32,compression="gzip"  )
        dset.attrs['valid_num']=0
        block_min, block_max,i_xyz = self.get_block_scope_from_k(block_k)
        dset.attrs['i_xyz'] = i_xyz
        dset.attrs['xyz_min'] = block_min
        dset.attrs['xyz_max'] = block_max
        #print('block %s min = %s  max = %s '%(dset_name,block_min,block_max))
        return dset
    def rm_invalid_data(self):
        for dset_name_i in self.sorted_h5f:
            dset_i = self.sorted_h5f[dset_name_i]
            valid_n = dset_i.attrs['valid_num']
            if dset_i.shape[0] > valid_n:
                #print('resizing block %s from %d to %d'%(dset_name_i,dset_i.shape[0],valid_n))
                dset_i.resize( (valid_n,dset_i.shape[1]) )

    def get_block_scope_from_k(self,block_k):
        i_xyz = self.block_index_to_ixyz(block_k)
        block_k = int( i_xyz[0]*self.block_dims_N[1]*self.block_dims_N[2] + i_xyz[1]*self.block_dims_N[2] + i_xyz[2] )
        block_min = i_xyz * self.block_stride + self.xyz_min_aligned
        block_max = block_min + self.block_step
        return block_min,block_max,i_xyz

    def check_xyz_scope_k(self,block_k):
        '''
        (1) anno-scope == scope_from_k
        (2) xyz data is in scope
        '''
        dset = self.sorted_h5f[str(block_k)]
        min_anno = dset.attrs['xyz_min']
        max_anno = dset.attrs['xyz_max']
        min_k,max_k,i_xyz = self.get_block_scope_from_k(block_k)

        e_min = min_anno-min_k
        e_max = max_anno-max_k
        e = np.linalg.norm(e_min) + np.linalg.norm(e_max)
        if e > 1e-5:
            print('block %d scope anno error! '%(block_k),'\nscope_k=\n',[min_k,max_k],'scope_anno=\n',[min_anno,max_anno])
            return False

        xyz = dset[:,0:3]
        xyz_max = xyz.max(axis=0)
        xyz_min = xyz.min(axis=0)
        if (max_k >= xyz_max).all() and (min_k <= xyz_min).all():
            #print('scope checked OK')
            return True
        else:
            if not (min_k <= xyz_min).all():
                print('\nmin check failed')
            if not (max_k >= xyz_max).all():
                print('\nmax check failed')
            print('scope_min=\n',min_k,'\nreal_min=\n',xyz_min)
            print('scope_max=\n',max_k,'\nreal_max=\n',xyz_max)
            print('stride=\n',self.block_stride,'\nstep=\n',self.block_step)
            return False
    def check_xyz_scope(self):
        step = int(self.total_block_N/20)+1
        Flag = True
        n=0
        for i,dset_n in enumerate(self.sorted_h5f):
            block_k = int(dset_n)
            if i%step == 0:
                flag = self.check_xyz_scope_k(block_k)
                if not flag:
                    Flag = False
                    print('dset: %s xyz scope check                   failed'%(dset_n))
                else:
                    n += 1
                    pass
                    #print('dset: %s xyz scope check passed'%(dset_n))
        if Flag:
            print('\nall %d dsets  xyz scope check passed\n'%(n))
        return Flag

    def check_equal_to_raw(self,raw_h5f):
        check_flag = True
        for k,block_k in enumerate(self.sorted_h5f):
            #print('checing block %s'%(block_k))
            dset_k = self.sorted_h5f[block_k]
            step = max(int(dset_k.shape[0]/30),1)
            for i in range(0,dset_k.shape[0],step):
                sorted_d_i = dset_k[i,0:-1]
                raw_k = int(dset_k[i,-1])
                if raw_k < 0 or raw_k > 16777215: # for float32, it is not accurate again
                    continue
                #raw_d_i = np.concatenate(  [raw_xyz_set[raw_k,:],raw_color_set[raw_k,:],raw_label_set[raw_k,:],raw_intensity_set[raw_k,:]] )
                raw_d_i = raw_h5f.get_all_dsets(raw_k,raw_k+1)
                error = raw_d_i - sorted_d_i
                err = np.linalg.norm( error )
                if err != 0:
                    check_flag = False
                    print('\nsorted error:raw_k=%d  block_k=%s,i=%d'%(raw_k,block_k,i))
                    print('raw_data = \n',raw_d_i,'\nsorted_data = \n',sorted_d_i)
                    break
                else:
                    pass
                    #print('equal check passed: block_k=%s,i=%d'%(block_k,i))
#            if flag_k:
#                    print('equal check passed: block_k=%s '%(block_k))
#            else:
#                    print('equal check failed: block_k=%s '%(block_k))
        return check_flag

    def append_to_dset(self,aim_block_k,source_dset,vacant_size=0,sample_method=None,sample_num=None):
        '''
        if append frequently to one dataset, vacant_size > 0 to avoid frequent resize
        '''
        source_N = source_dset.shape[0]
        if sample_method != None:
            sample_choice = self.sample(source_N,sample_num,sample_method)
            sample_choice = np.sort(sample_choice)
            new_row_N = sample_choice.size
        else:
            new_row_N = source_N

        aim_dset = self.get_blocked_dset(aim_block_k,vacant_size,self.total_num_channels)
        row_step = self.h5_num_row_1M * 10
        org_row_N = aim_dset.attrs['valid_num']
        aim_dset.resize((org_row_N+new_row_N+vacant_size,aim_dset.shape[1]))
        for k in range(0,new_row_N,row_step):
            end = min(k+row_step,new_row_N)
            if sample_method == None:
                aim_dset[org_row_N+k:org_row_N+end,:] = source_dset[k:end,:]
            else:
                choice_k = sample_choice[k:end]
                dset_buf = source_dset[choice_k.min():choice_k.max()+1,:]
                aim_dset[org_row_N+k:org_row_N+end,:] = dset_buf[choice_k-choice_k.min(),:]
            aim_dset.attrs['valid_num'] = end + org_row_N

    def generate_one_block_to_object(self,block_k,out_obj_file,IsLabelColor=False):
        row_step = self.h5_num_row_1M * 10
        dset_k = self.get_blocked_dset(block_k)
        row_N = dset_k.shape[0]
        for k in range(0,row_N,row_step):
            end = min(k+row_step,row_N)
            buf_k = dset_k[k:end,:]
            #buf_k[:,0:3] -= middle
            for j in range(0,buf_k.shape[0]):
                if not IsLabelColor:
                    str_j = 'v ' + ' '.join( ['%0.3f'%(d) for d in  buf_k[j,0:3]]) + ' \t'\
                    + ' '.join( ['%d'%(d) for d in  buf_k[j,3:6]]) + '\n'
                else:
                    label = buf_k[j,self.data_idxs['label'][0]]
                  #  if label == 0:
                  #      continue
                    label_color = Normed_H5f.g_label2color[label]
                    str_j = 'v ' + ' '.join( ['%0.3f'%(d) for d in  buf_k[j,0:3]]) + ' \t'\
                    + ' '.join( ['%d'%(d) for d in  label_color ]) + '\n'

                out_obj_file.write(str_j)

    def gen_file_obj(self,IsLabelColor=False):
        if self.file_name == None:
            print('set file_name (gen_file_obj)')
            return
        base_fn = os.path.basename(self.file_name)
        base_fn = os.path.splitext(base_fn)[0]
        folder_path = os.path.dirname(self.file_name)
        obj_folder = os.path.join(folder_path,base_fn)
        if not os.path.exists(obj_folder):
            os.makedirs(obj_folder)

        aim_scope = np.array([[-30,-30,-20],[20,20,50]])
        aim_scope = None
        n = 0
        last_rate = -20
        out_info_fn = os.path.join(obj_folder,'info.txt')
        with open(out_info_fn,'w') as info_f:
            for dset_name in self.sorted_h5f:
                row_N = self.sorted_h5f[dset_name].shape[0]

                min_i = self.sorted_h5f[dset_name].attrs['xyz_min']
                max_i = self.sorted_h5f[dset_name].attrs['xyz_max']
                if aim_scope == None:
                    IsInScope = True
                else:
                    IsInScope = (min_i > aim_scope[0,:]).all() and ( max_i < aim_scope[1,:]).all()
                if not IsInScope:
                    continue
                if IsLabelColor:
                    name_meta = 'labeled_'
                else:
                    name_meta = ''
                out_fn = os.path.join(obj_folder,name_meta+dset_name+'_'+str(row_N)+'.obj')
                with open(out_fn,'w') as out_f:
                    self.generate_one_block_to_object(dset_name,out_f,IsLabelColor)
                n += row_N
                rate = 100.0 * n / self.total_row_N
                if int(rate) % 2 == 0 and rate - last_rate > 3:
                    last_rate = rate
                    print('%0.2f%% generating file: %s'%(rate,os.path.basename(out_fn)) )

                info_str = 'dset: %s \tN= %d   \tmin=%s   \tmax=%s \n'%(dset_name,self.sorted_h5f[dset_name].shape[0], np.array_str(min_i), np.array_str(max_i)  )
                info_f.write(info_str)
                #print(info_str)
                #if rate > 30:
                    #break
    def extract_sub_area(self,sub_xyz_scope,sub_file_name):
        with h5py.File(sub_file_name,'w') as sub_h5f:
            sub_f = Sorted_H5f(sub_h5f,sub_file_name)

            sub_f.copy_root_summaryinfo_from_another(self.sorted_h5f,'sub')
            sub_f.set_step_stride(self.block_step,self.block_stride)
            for dset_name_i in self.sorted_h5f:
                xyz_min_i = self.sorted_h5f[dset_name_i].attrs['xyz_min']
                xyz_max_i = self.sorted_h5f[dset_name_i].attrs['xyz_max']
                if (xyz_min_i > sub_xyz_scope[0,:]).all() and (xyz_max_i < sub_xyz_scope[1,:]).all():
                    sub_f.get_blocked_dset(dset_name_i,0)
                    sub_f.append_to_dset(dset_name_i,self.sorted_h5f[dset_name_i])
            sub_f.add_total_row_block_N()

    def sample(self,org_N,sample_N,sample_method='random'):
        if sample_method == 'random':
            if org_N == sample_N:
                sample_choice = np.arange(sample_N)
            elif org_N > sample_N:
                sample_choice = np.random.choice(org_N,sample_N)
                self.reduced_num += org_N - sample_N
            else:
                #sample_choice = np.arange(org_N)
                new_samp = np.random.choice(org_N,sample_N-org_N)
                sample_choice = np.concatenate( (np.arange(org_N),new_samp) )
            #str = '%d -> %d  %d%%'%(org_N,sample_N,100.0*sample_N/org_N)
            #print(str)
        return sample_choice

    def file_sample(self,sample_num,sample_method,gen_norm=False,gen_obj=False):
        parts = os.path.splitext(self.file_name)
        sampled_filename =  parts[0]+'_'+sample_method+'_'+str(sample_num)+parts[1]
        print('start genrating sampled file: ',sampled_filename)
        ave_dset_num = self.total_row_N /  self.total_block_N
        print('ave_org_num = ',ave_dset_num)
        print('sample_num = %d   %d%%'%(sample_num,100.0*sample_num/ave_dset_num) )
        with h5py.File(sampled_filename,'w') as sampled_h5f:
            sampled_sh5f = Sorted_H5f(sampled_h5f,sampled_filename)
            sampled_sh5f.copy_root_summaryinfo_from_another(self.sorted_h5f,'sub')
            self.set_root_attr('sample_num',sample_num)
            self.set_root_attr('sample_method',sample_method)
            for i, k_str in enumerate( self.sorted_h5f ):
                dset_k = self.sorted_h5f[k_str]
                #if dset_k.shape[0] < sample_num*0.3:
                if dset_k.shape[0] < 100:
                    continue
                sampled_sh5f.append_to_dset(int(k_str),dset_k,vacant_size=0,\
                                            sample_method=sample_method,sample_num=sample_num)
            sampled_sh5f.add_total_row_block_N()
            print('reduced_num = %d  %d%%'%(sampled_sh5f.reduced_num,100.0*sampled_sh5f.reduced_num/self.total_row_N ))
            reduced_block_N = self.total_block_N - sampled_sh5f.total_block_N
            print('reduced block num = %d  %d%%'%(reduced_block_N,100*reduced_block_N/self.total_block_N))

            if gen_obj:
               sampled_sh5f.gen_file_obj()
            if gen_norm:
                sampled_sh5f.file_normalization()


    def get_sample_shape(self):
            for i,k_str in  enumerate(self.sorted_h5f):
                dset = self.sorted_h5f[k_str]
                return dset.shape

    def file_normalization(self):
        xyz_1norm_method = 'global'
        parts = os.path.splitext(self.file_name)
        normalized_filename =  parts[0]+'_'+xyz_1norm_method+'norm.nh5'
        print('start gen normalized file: ',normalized_filename)
        with h5py.File(normalized_filename,'w') as h5f:
            normed_h5f = Normed_H5f(h5f,normalized_filename)
            sample_point_n,raw_channels = self.get_sample_shape()
            normed_num_channels = raw_channels -1 - self.IS_CHECK + 3
            normed_h5f.create_dsets(self.total_block_N,sample_point_n,normed_num_channels)

            for i,k_str in  enumerate(self.sorted_h5f):
                normed_data_i,normed_label_i,raw_xyz_i = self.normalize_dset(k_str,xyz_1norm_method)
                normed_h5f.append_to_dset('data',normed_data_i)
                normed_h5f.append_to_dset('label',normed_label_i)
                normed_h5f.append_to_dset('raw_xyz',raw_xyz_i)
            normed_h5f.rm_invalid_data()
            print('normalization finished: data shape: %s'%(str(normed_h5f.data_set.shape)) )

    def merge_to_new_step(self,larger_stride,larger_step,more_actions_config=None):
        '''merge blocks of sorted raw h5f to get new larger step
        '''
        base_file_name = self.file_name
        tmp = base_file_name.rsplit('_stride_',1)[0]
        new_part = '_stride_' + str(larger_stride[0])+ '_step_' + str(larger_step[0])
     #   if larger_step[2] != larger_step[0]:
     #       if larger_step[2]>0:
     #           new_part += '_z' + str(larger_step[2])
     #       else:
     #           new_part += '_zall'

        new_name = tmp + new_part + '.sh5'
        print('new file: ',new_name)
        #if os.path.exists(new_name):
        #    print('already exists, skip')
        #    return
        with  h5py.File(base_file_name,'r') as base_h5f:
            with h5py.File(new_name,'w') as new_h5f:
                new_sh5f = Sorted_H5f(new_h5f,new_name)
                new_sh5f.copy_root_summaryinfo_from_another(base_h5f,'new_stride')
                new_sh5f.set_step_stride(larger_step,larger_stride)

                read_row_N = 0
                rate_last = -10
                print('%d rows and %d blocks to merge'%(self.total_row_N,self.total_block_N))
                for dset_name in  base_h5f:
                    block_i_base = int(dset_name)
                    base_dset_i = base_h5f[dset_name]
                    block_k_new_ls,i_xyz_new_ls = self.get_sub_block_ks(block_i_base,new_sh5f)

                    read_row_N += base_dset_i.shape[0]
                    rate = 100.0 * read_row_N / self.total_row_N
                    if int(rate)%10 < 1 and rate-rate_last>5:
                        rate_last = rate
                        print(str(rate),'%   ','  dset_name = ',dset_name, '  new_k= ',block_k_new_ls,'   id= ',os.getpid())
                        new_sh5f.sorted_h5f.flush()

                    for block_k_new in block_k_new_ls:
                        new_sh5f.append_to_dset(block_k_new,base_dset_i)
                    #if rate > 5:
                        #break
                if read_row_N != self.total_row_N:
                    print('ERROR!!!  total_row_N = %d, but only read %d'%( self.total_row_N,read_row_N))

                total_block_N = 0
                total_row_N = 0
                for total_block_N,dn in enumerate(new_sh5f.sorted_h5f):
                    total_row_N += new_sh5f.sorted_h5f[dn].shape[0]
                total_block_N += 1
                new_sh5f.set_root_attr('total_row_N',total_row_N)
                new_sh5f.set_root_attr('total_block_N',total_block_N)
                print('total_row_N = ',total_row_N)
                print('total_block_N = ',total_block_N)
                new_sh5f.sorted_h5f.flush()

                #new_sh5f.check_xyz_scope()

                if more_actions_config != None:
                    actions = more_actions_config['actions']
                    if 'obj_merged' in actions:
                        new_sh5f.gen_file_obj(True)
                        new_sh5f.gen_file_obj(False)
                    if 'sample_merged' in actions:
                        Is_gen_obj = 'obj_sampled_merged' in actions
                        Is_gen_norm = 'norm_sampled_merged' in actions
                        new_sh5f.file_sample(more_actions_config['sample_num'],more_actions_config['sample_method'],\
                                            gen_norm=Is_gen_norm,gen_obj = Is_gen_obj)



class Normed_H5f():
    '''
    (1)normed
    (2) 'data' data_set store all normalized data, shape: N*(H*W)*C, like [N,4096,9]
        'label'
        'pred_label'
        'raw_xyz'
    '''
    # -----------------------------------------------------------------------------
    # CONSTANTS
    # -----------------------------------------------------------------------------
    g_label2class_dic = {}
    g_label2class_dic['ETH'] = {0: 'unlabeled points', 1: 'man-made terrain', 2: 'natural terrain',\
                     3: 'high vegetation', 4: 'low vegetation', 5: 'buildings', \
                     6: 'hard scape', 7: 'scanning artefacts', 8: 'cars'}

    g_label2class_dic['STANFORD_INDOOR3D'] = \
                    {0:'ceiling',
                    1:'floor',
                    2:'wall',
                    3:'beam',
                    4:'column',
                    5:'window',
                    6:'door',
                    7:'table',
                    8:'chair',
                    9:'sofa',
                    10:'bookcase',
                    11:'board',
                    12:'clutter'}
    g_label2color_dic = {}
    g_label2color_dic['ETH'] = \
                    {0:	[0,0,0],
                     1:	[0,0,255],
                     2:	[0,255,255],
                     3: [255,255,0],
                     4: [255,0,255],
                     6: [0,255,0],
                     7: [170,120,200],
                     8: [255,0,0],
                     5:[10,200,100]}
    g_label2color_dic['STANFORD_INDOOR3D'] = \
                    {0:	[0,0,0],
                     1:	[0,0,255],
                     2:	[0,255,255],
                     3: [255,255,0],
                     4: [255,0,255],
                     10: [100,100,255],
                     6: [0,255,0],
                     7: [170,120,200],
                     8: [255,0,0],
                     9: [200,100,100],
                     5:[10,200,100],
                     11:[200,200,200],
                     12:[50,50,50],
                     13:[200,200,100]}
    g_label2class = g_label2class_dic[DATASET_NAME]
    g_label2color = g_label2color_dic[DATASET_NAME]


    g_class2label = {cls:label for label,cls in g_label2class.iteritems()}
    g_class2color = {}
    for i in g_label2class:
        cls = g_label2class[i]
        g_class2color[cls] = g_label2color[i]
    NUM_CLASSES = len(g_label2class)
    #g_easy_view_labels = [7,8,9,10,11,1]
    #g_is_labeled = True

    ## normed data channels
    data_elements = ['xyz_midnorm','color_1norm','xyz_1norm','intensity_1norm']
    elements_idxs = {data_elements[0]:range(0,3),data_elements[1]:range(3,6),\
                     data_elements[2]:range(6,9),data_elements[3]:range(9,10)}

    def __init__(self,h5f,file_name):
        self.h5f = h5f
        self.file_name = file_name

        self.dataset_names = ['data','label','raw_xyz','pred_label']
        for dn in self.dataset_names:
            if dn in h5f:
                setattr(self,dn+'_set', h5f[dn])
    @staticmethod
    def show_all_colors():
        from PIL import Image
        for label,color in Normed_H5f.g_label2color.iteritems():
            if label < len(Normed_H5f.g_label2class):
                cls = Normed_H5f.g_label2class[label]
            else:
                cls = 'empty'
            data = np.zeros((512,512,3),dtype=np.uint8)
            color_ = np.array(color,dtype=np.uint8)
            data += color_
            img = Image.fromarray(data,'RGB')
            img.save('colors/'+str(label)+'_'+cls+'.png')
            img.show()

    def label2color(self,label):
        assert( label in self.g_label2color )
        return self.g_label2color[label]

    def get_data_shape(self):
        dset = self.h5f['data']
        return dset.shape

    def create_dsets(self,total_block_N,sample_num,num_channels):
        chunks_n = 4
        data_set = self.h5f.create_dataset( 'data',shape=(total_block_N,sample_num,num_channels),\
                maxshape=(None,sample_num,num_channels),dtype=np.float32,compression="gzip",\
                chunks = (chunks_n,sample_num,num_channels)  )
        label_set = self.h5f.create_dataset( 'label',shape=(total_block_N,sample_num),\
                maxshape=(None,sample_num),dtype=np.int16,compression="gzip",\
                chunks = (chunks_n,sample_num)  )

        # record the original xyz for gen obj
        raw_xyz_set = self.h5f.create_dataset( 'raw_xyz',shape=(total_block_N,sample_num,3),\
                maxshape=(None,sample_num,3),dtype=np.float32,compression="gzip",\
                chunks = (chunks_n,sample_num,3)  )
        # predicted label
        pred_label_set = self.h5f.create_dataset( 'pred_label',shape=(total_block_N,sample_num),\
                maxshape=(None,sample_num),dtype=np.int16,compression="gzip",\
                chunks = (chunks_n,sample_num)  )
        pred_label_set[:] = -1

        data_set.attrs['elements'] = self.data_elements
        for ele in self.data_elements:
            data_set.attrs[ele] = self.elements_idxs[ele]
        data_set.attrs['valid_num'] = 0
        label_set.attrs['valid_num'] = 0
        raw_xyz_set.attrs['valid_num'] = 0
        pred_label_set.attrs['valid_num'] = 0
        self.data_set = data_set
        self.label_set  =label_set
        self.raw_xyz_set = raw_xyz_set
        self.pred_label_set = pred_label_set

    def create_areano_dset(self,total_block_N,sample_num):
        chunks_n = 4
        area_no_set = self.h5f.create_dataset( 'area_no',shape=(total_block_N,sample_num),\
                maxshape=(None,sample_num),dtype=np.int16,compression="gzip",\
                chunks = (chunks_n,sample_num)  )
        area_no_set.attrs['valid_num'] = 0

    def append_to_dset(self,dset_name,data_i,vacant_size=0):
        dset = self.h5f[dset_name]
        valid_num = dset.attrs['valid_num']
        if data_i.ndim == len(dset.shape) -1:
            for i in range(1,len(dset.shape)):
                assert(dset.shape[i] == data_i.shape[i-1]), "in Normed_H5f.append_to_dset: data shape not match dataset"
            new_valid_num = valid_num + 1
            #print('append 2d to 3d')
        else:
            assert(dset.shape[1:] == data_i.shape[1:]), "in Normed_H5f.append_to_dset: data shape not match dataset"
            new_valid_num = valid_num + data_i.shape[0]
            print('%s  %d -> %d'%(dset_name,valid_num,new_valid_num) )

        if new_valid_num > dset.shape[0]:
            dset.resize( (new_valid_num + vacant_size,)+dset.shape[1:] )
        dset[valid_num : new_valid_num,...] = data_i
        dset.attrs['valid_num'] = new_valid_num
        self.h5f.flush()

    def set_dset_value(self,dset_name,data_i,start_idx,end_idx):
        if dset_name not in self.h5f:
            return
        dset = self.h5f[dset_name]
        if dset.shape[0] < end_idx:
            dset.resize( (end_idx,) + dset.shape[1:] )
        if data_i.shape[1] < dset.shape[1]:
            dset[start_idx:end_idx,data_i.shape[1]:] = -1
        dset[start_idx:end_idx,0:data_i.shape[1]] = data_i
        if dset.attrs['valid_num'] < end_idx:
            dset.attrs['valid_num'] = end_idx

    def merge_file(self,another_file_name):
        # merge all the data from another_file intto self
        with h5py.File(another_file_name,'r') as f:
            ano_normed_h5f = Normed_H5f(f,another_file_name)
            for dset_name in ano_normed_h5f.h5f:
                self.append_to_dset(dset_name,ano_normed_h5f.h5f[dset_name])
                # set area no
            if DATASET_NAME == 'STANFORD_INDOOR3D':
                base_name = os.path.basename(another_file_name)
                tmp = base_name.split('Area_')[1].split('_')[0]
                area_no = int(tmp)

                num_blocks,num_sample = ano_normed_h5f.h5f['label'].shape
                area_data = np.ones((num_blocks,num_sample)) * area_no
                self.append_to_dset('area_no',area_data)

    def rm_invalid_data(self):
        for dset_name_i in self.h5f:
            dset_i = self.h5f[dset_name_i]
            valid_n = dset_i.attrs['valid_num']
            if dset_i.shape[0] > valid_n:
                #print('resizing block %s from %d to %d'%(dset_name_i,dset_i.shape[0],valid_n))
                dset_i.resize( (valid_n,)+dset_i.shape[1:] )


    def Get_file_accuracies(self,IsWrite=False,out_path=None):
        # get the accuracy of each file by the pred data in hdf5
        if self.pred_label_set.shape[0] != self.label_set.shape[0]:
            return ''
        class_num = len(self.g_class2label)
        class_TP = np.ones(shape=(class_num))* class_num
        class_FN = np.ones(shape=(class_num))* class_num
        class_FP = np.ones(shape=(class_num))* class_num
        total_correct = 0.0
        total_seen = 0.0

        for j in range(0,self.raw_xyz_set.shape[0]):
            xyz_block = self.raw_xyz_set[j,:]
            label_gt = self.label_set[j,:]
            label_pred = self.pred_label_set[j,:]
            for i in range(xyz_block.shape[0]):
                # calculate accuracy
                total_seen += 1
                if (label_gt[i]==label_pred[i]):
                    total_correct += 1
                    class_TP[label_gt[i]] += 1
                else:
                    class_FN[label_gt[i]] += 1
                    class_FP[label_pred[i]] += 1
        # calculate accuracy
        class_precision = [class_TP[c]/(class_TP[c]+class_FP[c]) for c in range(class_num)]
        class_recall = [class_TP[c]/(class_TP[c]+class_FN[c]) for c in range(class_num)]
        class_IOU = [class_TP[c]/(class_TP[c]+class_FP[c]+class_FN[c]) for c in range(class_num)]
        total_accu = total_correct  / total_seen
        def get_str(ls):
            return ',\t'.join(['%0.3f'%v for v in ls])
        acc_str = 'total accuracy:  %3f    N = %3f M\n'%(total_accu,self.raw_xyz_set.size/1000000.0)
        acc_str += '\t     '+',  '.join([c for c in self.g_class2label])+'\n'
        acc_str += 'class_pre:   '+get_str(class_precision)+'\n'
        acc_str += 'class_rec:   '+get_str(class_recall)+'\n'
        acc_str += 'class_IOU:   '+get_str(class_IOU)+'\n'

        if IsWrite:
            base_fn = os.path.basename(self.file_name)
            base_fn = os.path.splitext(base_fn)[0]
            folder_path = os.path.dirname(self.file_name)
            if out_path == None:
                obj_folder = os.path.join(folder_path,'obj_file',base_fn)
            else:
                obj_folder = os.path.join(out_path,base_fn)
            print('obj_folder=',obj_folder)
            if not os.path.exists(obj_folder):
                os.makedirs(obj_folder)
            accuracy_fn = os.path.join(obj_folder,'accuracies.txt')
            with open(accuracy_fn,'w') as accuracy_f:
                    accuracy_f.write(acc_str)
        return acc_str


    def gen_gt_pred_obj_examples(self,config_flag = ['None'],out_path=None):
        #{'ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter'}
        config_flag = ['building_6_no_ceiling']
        def get_config(config_flag):
            if config_flag =='None':
                xyz_cut_rate=None
                show_categaries=None
            if config_flag =='only_ceiling':
                xyz_cut_rate=None
                show_categaries=['ceiling']
            if config_flag =='yZ':
                xyz_cut_rate=[0,0.06,0.93]
                show_categaries=None
            if config_flag =='XZ':
                xyz_cut_rate=[0.95,0,0.93]
                show_categaries=None
            if config_flag =='building_7':
                xyz_cut_rate=None
                show_categaries=['ceiling','floor','wall','beam','column','window','door']
            if config_flag =='building_6_no_ceiling':
                xyz_cut_rate=None
                show_categaries=['floor','wall','beam','column','window','door']
            return xyz_cut_rate,show_categaries
        for flag in config_flag:
            xyz_cut_rate,show_categaries = get_config(flag)
            #self.gen_gt_pred_obj(out_path,xyz_cut_rate,show_categaries,visu_flag=str(flag))
            self.Get_file_accuracies(IsWrite=True)

    def gen_gt_pred_obj(self,out_path=None,xyz_cut_rate=None,show_categaries=None,visu_flag=None):
        '''
            (1)xyz_cut_rate:
                # when rate < 0.5: cut small
                # when rate >0.5: cut big
            (2) show_categaries:  ['ceiling']
                the categaries to show, if None  show all
        '''
        if show_categaries != None:
            show_categaries = [self.g_class2label[c] for c in show_categaries]
        if self.pred_label_set.shape[0] ==0:
            print('File: %s \n   has no pred data'%(self.file_name))
            return
        base_fn = os.path.basename(self.file_name)
        base_fn = os.path.splitext(base_fn)[0]
        folder_path = os.path.dirname(self.file_name)
        if out_path == None:
            obj_folder = os.path.join(folder_path,'obj_file',base_fn)
            if visu_flag != None:
                obj_folder = os.path.join(obj_folder,visu_flag)
        else:
            obj_folder = os.path.join(out_path,base_fn)
        print('obj_folder=',obj_folder)
        if not os.path.exists(obj_folder):
            os.makedirs(obj_folder)

        raw_obj_fn = os.path.join(obj_folder,'raw.obj')
        raw_colored_obj_fn = os.path.join(obj_folder,'raw_colored.obj')
        gt_obj_fn = os.path.join(obj_folder,'gt.obj')
        pred_obj_fn = os.path.join(obj_folder,'pred.obj')
        dif_obj_fn = os.path.join(obj_folder,'dif.obj')
        correct_obj_fn = os.path.join(obj_folder,'correct.obj')
        correct_num = 0
        pred_num = 0
        file_size = self.raw_xyz_set.shape[0] * self.raw_xyz_set.shape[1]

        if xyz_cut_rate != None:
            # when rate < 0.5: cut small
            # when rate >0.5: cut big
            xyz_max = np.array([np.max(self.raw_xyz_set[:,:,i]) for i in range(3)])
            xyz_min = np.array([np.min(self.raw_xyz_set[:,:,i]) for i in range(3)])
            xyz_scope = xyz_max - xyz_min
            xyz_thres = xyz_scope * xyz_cut_rate + xyz_min
            print('xyz_thres = ',str(xyz_thres))
        cut_num = 0

        with open(gt_obj_fn,'w') as gt_f, open(raw_obj_fn,'w') as raw_f, open(raw_colored_obj_fn,'w') as raw_colored_f:
          with open(pred_obj_fn,'w') as pred_f,open(dif_obj_fn,'w') as dif_f:
            with open(correct_obj_fn,'w') as correct_f:
                for j in range(0,self.raw_xyz_set.shape[0]):
                    xyz_block = self.raw_xyz_set[j,:]
                    label_gt = self.label_set[j,:]
                    if j < self.pred_label_set.shape[0]:
                        IsGenPred = True
                        label_pred = self.pred_label_set[j,:]
                    else:
                        IsGenPred = False
                    for i in range(xyz_block.shape[0]):

                        # cut parts by xyz or label
                        is_cut_this_point = False
                        if show_categaries!=None and label_gt[i] not in show_categaries:
                            is_cut_this_point = True
                        elif xyz_cut_rate!=None:
                            for xyz_j in range(3):
                                if (xyz_cut_rate[xyz_j] >0.5 and xyz_block[i,xyz_j] > xyz_thres[xyz_j]) or \
                                    (xyz_cut_rate[xyz_j]<=0.5 and xyz_block[i,xyz_j] < xyz_thres[xyz_j]):
                                    is_cut_this_point =  True
                        if is_cut_this_point:
                            cut_num += 1
                            continue

                        color_gt = self.label2color( label_gt[i] )
                        str_xyz = 'v ' + ' '.join( ['%0.3f'%(d) for d in  xyz_block[i,:] ])
                        raw_f.write(str_xyz+'\n')
                        str_xyz = str_xyz + ' \t'
                        str_raw_color = ' '.join( ['%d'%(d) for d in  256*self.data_set[j,i,self.elements_idxs['color_1norm']]]) + '\n'
                        raw_colored_f.write(str_xyz+str_raw_color)
                        str_color_gt = ' '.join( ['%d'%(d) for d in  color_gt]) + '\n'
                        str_gt = str_xyz + str_color_gt
                        gt_f.write( str_gt )

                        if IsGenPred and label_pred[i] in self.g_label2color:
                            color_pred = self.label2color( label_pred[i] )
                            str_color_pred = ' '.join( ['%d'%(d) for d in  color_pred]) + '\n'
                            str_pred = str_xyz + str_color_pred
                            pred_f.write( str_pred )
                            if label_gt[i] != label_pred[i]:
                                dif_f.write(str_pred)
                            else:
                                correct_f.write(str_pred)
                                correct_num += 1
                            pred_num += 1
                    if j%10 ==0: print('batch %d / %d'%(j,self.raw_xyz_set.shape[0]))


                print('gen gt obj file (%d): \n%s'%(file_size,gt_obj_fn) )
                if pred_num > 0:
                     print('gen pred obj file (%d,%f): \n%s '%(pred_num,1.0*pred_num/file_size,pred_obj_fn) )
                     print('gen correct obj file (%d,%f),: \n%s '%(correct_num,1.0*correct_num/pred_num,correct_obj_fn) )
                print('gen dif obj file: ',pred_obj_fn)
                print('cut roof ponit num = %d, xyz_cut_rate = %s'%(cut_num,str(xyz_cut_rate)) )

def Write_all_file_accuracies(normed_h5f_file_list=None,out_path=None):
    if normed_h5f_file_list == None:
        normed_h5f_file_list = glob.glob( GLOBAL_PARA.stanford_indoor3d_globalnormedh5_stride_0d5_step_1_4096 +
                            '/Area_6*' )
    if out_path == None: out_path = os.path.join(GLOBAL_PARA.stanford_indoor3d_globalnormedh5_stride_0d5_step_1_4096,
                                    'obj_file')
    all_acc_fn = os.path.join(out_path,'all_file_accuracies.txt')
    with open(all_acc_fn,'w') as all_acc_f:
        for i,fn in enumerate(normed_h5f_file_list):
            h5f = h5py.File(fn,'r')
            norm_h5f = Normed_H5f(h5f,fn)
            acc_str = norm_h5f.Get_file_accuracies(IsWrite=False, out_path = out_path)
            if acc_str != '':
                all_acc_f.write('File: '+os.path.basename(fn)+'\n')
                all_acc_f.write(acc_str+'\n')

class SortSpace():
    '''
    (1) Do sort: from "Norm_H5f" to "Sorted_H5f"
    '''
    def __init__(self,raw_file_list,block_step_xyz=[0.5,0.5,0.5]):
        self.Do_sort_to_blocks(raw_file_list,block_step_xyz)

    def Do_sort_to_blocks(self,raw_file_list,block_step_xyz = [0.5,0.5,0.5]):
        IsMulti = False
        if not IsMulti:
            for fn in raw_file_list:
                self.sort_to_blocks(fn,block_step_xyz)
        else:
            #pool = mp.Pool( max(mp.cpu_count()/2,1) )
            print('cpu_count= ',mp.cpu_count())
            pool = mp.Pool()
            for fn in raw_file_list:
                pool.apply_async(self.sort_to_blocks(fn,block_step_xyz))
            pool.close()
            pool.join()

    def sort_to_blocks(self,file_name,block_step_xyz=[1,1,1]):
        '''
        split th ewhole scene to space sorted small blocks
        The whole scene is a group. Each block is one dataset in the group.
        The block attrs represents the field.
        '''
        print('start sorting file to blocks: %s'%file_name)
        block_step = np.array( block_step_xyz )
        print('block step = ',block_step)
        self.row_num_limit = None

        tmp = block_step[0]
        if tmp % 1 ==0:
            tmp = int(tmp)
        fn = rm_file_name_midpart(file_name,'_intensity_rgb')
        fn = rm_file_name_midpart(fn,'_xyz')
        blocked_file_name = os.path.splitext(fn)[0]+'_stride_'+str(tmp)+'_step_'+str(tmp)+'.sh5'
        with h5py.File(blocked_file_name,'w') as h5f_blocked:
            with h5py.File(file_name,'r') as h5_f:
                self.raw_h5f = Raw_H5f(h5_f,file_name)
                self.s_h5f = Sorted_H5f(h5f_blocked,blocked_file_name)

                self.s_h5f.copy_root_geoinfo_from_raw( self.raw_h5f )
                self.s_h5f.set_step_stride(block_step,block_step)

                #self.row_num_limit = int(self.raw_h5f.total_row_N/1000)

                row_step = GLOBAL_PARA.h5_num_row_1M*8
                sorted_buf_dic = {}
                raw_row_N = self.raw_h5f.xyz_dset.shape[0]

                for k in range(0,raw_row_N,row_step):
                    end = min(k+row_step,raw_row_N)
                    _,data_name_list = self.raw_h5f.get_total_num_channels_name_list()
                    raw_buf = np.zeros((end-k,self.s_h5f.total_num_channels))
                    for dn in data_name_list:
                        raw_buf[:,self.s_h5f.data_idxs[dn] ] = self.raw_h5f.raw_h5f[dn][k:end,:]
                    if self.s_h5f.IS_CHECK:
                        if end < 16777215: # this is the largest int float32 can acurately present
                            org_row_index = np.arange(k,end)
                        else:
                            org_row_index = -1
                        raw_buf[:,self.s_h5f.data_idxs['org_row_index'][0]] = org_row_index

                    sorted_buf_dic={}
                    self.sort_buf(raw_buf,k,sorted_buf_dic)

                    self.h5_write_buf(sorted_buf_dic)

                    if int(k/row_step) % 1 == 0:
                        print('%%%.1f  line[ %d:%d ] block_N = %d'%(100.0*end/self.raw_h5f.total_row_N, k,end,len(sorted_buf_dic)))
                         #print('line: [%d,%d] blocked   block_T=%f s, read_T=%f ms, cal_t = %f ms, write_t= %f ms'%\
                               #(k,end,time.time()-t0_k,(t1_k-t0_k)*1000,(t2_1_k-t2_0_k)*1000, (t2_2_k-t2_1_k)*1000 ))
                    if hasattr(self,'row_num_limit') and self.row_num_limit!=None and  end>=self.row_num_limit:
                    #if k /row_step >3:
                        print('break read at k= ',end)
                        break

                total_row_N,total_block_N = self.s_h5f.add_total_row_block_N()

                if total_row_N != self.raw_h5f.total_row_N:
                    print('ERROR: blocked total_row_N= %d, raw = %d'%(total_row_N,self.raw_h5f.total_row_N))
                print('total_block_N = ',total_block_N)

                if self.s_h5f.IS_CHECK:
                    check = self.s_h5f.check_equal_to_raw(self.raw_h5f) & self.s_h5f.check_xyz_scope()
                    print('overall check of equal and scope:')
                    if check:
                        print('both passed')
                    else:
                        print('somewhere check failed')
                #self.s_h5f.show_summary_info()

    def sort_buf(self,raw_buf,buf_start_k,sorted_buf_dic):
        #t0 = time.time()
        IsMulti = False
        if IsMulti:
            block_ks = self.get_block_index_multi(raw_buf)
        else:
            block_ks = np.zeros(raw_buf.shape[0],np.int64)
            for j in range(raw_buf.shape[0]):
                block_ks[j] = self.s_h5f.xyz_to_block_index(raw_buf[j,0:3])

        #t1 = time.time()
        for i in range(raw_buf.shape[0]):
            block_k = block_ks[i]
            row = raw_buf[i,:].reshape(1,-1)
            if not block_k in sorted_buf_dic:
                sorted_buf_dic[block_k]=[]
            sorted_buf_dic[block_k].append(row)
        #t2 = time.time()
        #print('t1 = %d ms, t2 = %d ms'%( (t1-t0)*1000,(t2-t1)*1000 ))

    def h5_write_buf(self,sorted_buf_dic):
        for key in sorted_buf_dic:
            sorted_buf_dic[key] = np.concatenate(sorted_buf_dic[key],axis=0)
        for block_k in sorted_buf_dic:
            self.s_h5f.append_to_dset(block_k,sorted_buf_dic[block_k],vacant_size=GLOBAL_PARA.h5_num_row_1M)

#            dset_k =  self.s_h5f.get_blocked_dset(block_k)
#            valid_n = dset_k.attrs['valid_num']
#            new_valid_n = valid_n + sorted_buf_dic[block_k].shape[0]
#            while dset_k.shape[0] < new_valid_n:
#                dset_k.resize(( dset_k.shape[0]+self.h5_num_row_1M,dset_k.shape[1]))
#            dset_k[valid_n:new_valid_n,:] = sorted_buf_dic[block_k]
#            dset_k.attrs['valid_num'] = new_valid_n
        self.s_h5f.rm_invalid_data()
        self.s_h5f.sorted_h5f.flush()

    def get_block_index_multi(self,raw_buf):
        block_ks = mp.Array('i',raw_buf.shape[0])
        num_workers = 2
        step = int(raw_buf.shape[0]/num_workers)
        pool = []
        for i in range(0,raw_buf.shape[0],step):
            end = min( (i+1)*step, raw_buf.shape[0])
            p = mp.Process(target=self.get_block_index_subbuf,args=(raw_buf[i:end,0:3],block_ks,i) )
            p.start()
            pool.append(p)
        for p in pool:
            p.join()
        return block_ks

    def get_block_index_subbuf(self,sub_buf_xyz,block_ks,i_start):
        for i in range(sub_buf_xyz.shape[0]):
            block_ks[i+i_start] = self.s_h5f.xyz_to_block_index(sub_buf_xyz[i,0:3])


#-------------------------------------------------------------------------------
# provider for training and testing
#------------------------------------------------------------------------------
class Net_Provider():
    '''
    (1) provide data for training
    (2) load file list to list of Norm_H5f[]
    '''
    # input normalized h5f files
    # normed_h5f['data']: [blocks*block_num_point*num_channel],like [1000*4096*9]
    # one batch would contain sevel(batch_size) blocks,this will be set out side
    # provider with train_start_idx and test_start_idx


    def __init__(self,all_file_list,only_evaluate,eval_fn_glob,\
                 NUM_POINT_OUT=None,no_color_1norm = False,no_intensity_1norm = True,\
                 train_num_block_rate=1,eval_num_block_rate=1 ):
        train_file_list,eval_file_list = self.split_train_eval_file_list\
                            (all_file_list,only_evaluate,eval_fn_glob)
        self.no_color_1norm = no_color_1norm
        self.no_intensity_1norm = no_intensity_1norm
        self.NUM_POINT_OUT = NUM_POINT_OUT
        if only_evaluate:
            open_type = 'a' # need to write pred labels
        else:
            open_type = 'r'
        train_file_N = len(train_file_list)
        eval_file_N = len(eval_file_list)
        self.g_file_N = train_file_N + eval_file_N
        self.normed_h5f_file_list =  normed_h5f_file_list = train_file_list + eval_file_list

        self.norm_h5f_L = []
        # global: within the whole train/test dataset  (several files)
        # record the start/end row idx  of each file to help search data from
        # all files
        # [start_global_row_idxs,end_global__idxs]
        self.g_block_idxs = np.zeros((self.g_file_N,2),np.int32)
        self.eval_global_start_idx = None
        for i,fn in enumerate(normed_h5f_file_list):
            assert(os.path.exists(fn))
            h5f = h5py.File(fn,open_type)
            norm_h5f = Normed_H5f(h5f,fn)
            self.norm_h5f_L.append( norm_h5f )
            self.g_block_idxs[i,1] = self.g_block_idxs[i,0] + norm_h5f.data_set.shape[0]
            if i<self.g_file_N-1:
                self.g_block_idxs[i+1,0] = self.g_block_idxs[i,1]

        self.eval_global_start_idx = self.g_block_idxs[train_file_N,0]
        if train_file_N > 0:
            self.train_num_blocks = self.g_block_idxs[train_file_N-1,1]
        else: self.train_num_blocks = 0
        self.eval_num_blocks = self.g_block_idxs[-1,1] - self.train_num_blocks

        # use only part of the data to test code:
        if train_num_block_rate!=1 or eval_num_block_rate!=1:
            self.get_data_label_shape()
            print('whole train data shape: %s'%(str(self.train_data_shape)))
            print('whole eval data shape: %s'%(str(self.eval_data_shape)))
            # train: use the front part
            self.train_num_blocks = int( self.train_num_blocks * train_num_block_rate )
            if not only_evaluate:
                self.train_num_blocks = max(self.train_num_blocks,2)
            new_eval_num_blocks = int( max(2,self.eval_num_blocks * eval_num_block_rate) )
            # eval:use the back part, so train_file_list and eval_file_list can be
            # the same
            self.eval_global_start_idx += self.eval_num_blocks - new_eval_num_blocks
            self.eval_num_blocks = new_eval_num_blocks

        self.get_data_label_shape()
        #self.test_tmp()

    def split_train_eval_file_list(self,all_file_list,only_evaluate,eval_fn_glob=None):
        if only_evaluate:
            train_file_list = []
            eval_file_list = all_file_list
        else:
            if eval_fn_glob == None:
                eval_fn_glob = 'Area_6'
            train_file_list = []
            eval_file_list = []
            for fn in all_file_list:
                if fn.find(eval_fn_glob) > 0:
                    eval_file_list.append(fn)
                else:
                    train_file_list.append(fn)
        log_str = '\ntrain file list (n=%d) = \n%s\n\n'%(len(train_file_list),train_file_list[-2:])
        log_str += 'eval file list (n=%d) = \n%s\n\n'%(len(eval_file_list),eval_file_list[-2:])
        self.file_list_logstr = log_str
        return train_file_list,eval_file_list
    def get_data_label_shape(self):
        data_batches,label_batches = self.get_train_batch(0,1)
        self.train_data_shape = list(data_batches.shape)
        self.train_data_shape[0] = self.train_num_blocks
        self.num_channels = self.train_data_shape[2]
        self.eval_data_shape = list(data_batches.shape)
        self.eval_data_shape[0] = self.eval_num_blocks

    def test_tmp(self):
        s = 0
        e = 1
        train_data,train_label = self.get_train_batch(s,e)
        eval_data,eval_label = self.get_eval_batch(s,e)
        print('train:\n',train_data[0,0,:])
        print('eval:\n',eval_data[0,0,:])
        print('err=\n',train_data[0,0,:]-eval_data[0,0,:])


    def __exit__(self):
        print('exit Net_Provider')
        for norm_h5f in self.norm_h5f:
            norm_h5f.h5f.close()

    def global_idx_to_local(self,g_start_idx,g_end_idx):
        assert(g_start_idx>=0 and g_start_idx<=self.g_block_idxs[-1,1])
        assert(g_end_idx>=0 and g_end_idx<=self.g_block_idxs[-1,1])
        for i in range(self.g_file_N):
            if g_start_idx >= self.g_block_idxs[i,0] and g_start_idx < self.g_block_idxs[i,1]:
                start_file_idx = i
                local_start_idx = g_start_idx - self.g_block_idxs[i,0]
                for j in range(i,self.g_file_N):
                    if g_end_idx > self.g_block_idxs[j,0] and g_end_idx <= self.g_block_idxs[j,1]:
                        end_file_idx = j
                        local_end_idx = g_end_idx - self.g_block_idxs[j,0]

        return start_file_idx,end_file_idx,local_start_idx,local_end_idx

    def set_pred_label_batch(self,pred_label,g_start_idx,g_end_idx):
        start_file_idx,end_file_idx,local_start_idx,local_end_idx = \
            self.global_idx_to_local(g_start_idx,g_end_idx)
        pred_start_idx = 0
        for f_idx in range(start_file_idx,end_file_idx+1):
            if f_idx == start_file_idx:
                start = local_start_idx
            else:
                start = 0
            if f_idx == end_file_idx:
                end = local_end_idx
            else:
                end = self.norm_h5f_L[f_idx].label_set.shape[0]
            n = end-start
            self.norm_h5f_L[f_idx].set_dset_value('pred_label',\
                pred_label[pred_start_idx:pred_start_idx+n,:],start,end)
            pred_start_idx += n
        self.norm_h5f_L[f_idx].h5f.flush()


    def get_global_batch(self,g_start_idx,g_end_idx):
        start_file_idx,end_file_idx,local_start_idx,local_end_idx = \
            self.global_idx_to_local(g_start_idx,g_end_idx)

        data_ls = []
        label_ls = []
        for f_idx in range(start_file_idx,end_file_idx+1):
            if f_idx == start_file_idx:
                start = local_start_idx
            else:
                start = 0
            if f_idx == end_file_idx:
                end = local_end_idx
            else:
                end = self.norm_h5f_L[f_idx].label_set.shape[0]

            data_i = self.norm_h5f_L[f_idx].data_set[start:end,:,:]
            label_i = self.norm_h5f_L[f_idx].label_set[start:end,:]
            data_ls.append(data_i)
            label_ls.append(label_i)
        data_batches = np.concatenate(data_ls,0)
        data_batches = self.extract_channels(data_batches)
        label_batches = np.concatenate(label_ls,0)
        data_batches,label_batches = self.sample(data_batches,label_batches,self.NUM_POINT_OUT)

     #   print('\nin global')
     #   print('file_start = ',start_file_idx)
     #   print('file_end = ',end_file_idx)
     #   print('local_start = ',local_start_idx)
     #   print('local end = ',local_end_idx)
     #   #print('data = \n',data_batches[0,:])

        return data_batches,label_batches

    def sample(self,data_batches,label_batches,NUM_POINT_OUT):
        NUM_POINT_IN = data_batches.shape[1]
        if NUM_POINT_OUT == None:
            NUM_POINT_OUT = NUM_POINT_IN
        if NUM_POINT_IN != NUM_POINT_OUT:
            sample_choice = GLOBAL_PARA.sample(NUM_POINT_IN,NUM_POINT_OUT,'random')
            data_batches = data_batches[:,sample_choice,...]
            label_batches = label_batches[:,sample_choice]
        return data_batches,label_batches

    def extract_channels(self,data_batches):
        # extract the data types to be trained
        # xyz_1norm xyz_midnorm color_1norm intensity_1norm
        COLOR_IDXS = Normed_H5f.elements_idxs['color_1norm']
        INTENSITY_IDX = Normed_H5f.elements_idxs['intensity_1norm']
        delete_idxs = []
        if self.no_color_1norm:
            delete_idxs += COLOR_IDXS
        if self.no_intensity_1norm:
            delete_idxs += INTENSITY_IDX
        data_batches = np.delete(data_batches,delete_idxs,2)
        return data_batches

    def get_train_batch(self,train_start_batch_idx,train_end_batch_idx):
        # all train files are before eval files
        g_start_batch_idx = train_start_batch_idx
        g_end_batch_idx = train_end_batch_idx
        return self.get_global_batch(g_start_batch_idx,g_end_batch_idx)

    def get_eval_batch(self,eval_start_batch_idx,eval_end_batch_idx):
        g_start_batch_idx = eval_start_batch_idx + self.eval_global_start_idx
        g_end_batch_idx = eval_end_batch_idx + self.eval_global_start_idx
        return self.get_global_batch(g_start_batch_idx,g_end_batch_idx)

    def gen_gt_pred_objs(self,visu_fn_glob='The glob for file to be visualized',obj_dump_dir=None):
        for k,norm_h5f in enumerate(self.norm_h5f_L):
            if norm_h5f.file_name.find(visu_fn_glob) > 0:
                norm_h5f.gen_gt_pred_obj( obj_dump_dir )

    def write_file_accuracies(self,obj_dump_dir=None):
        Write_all_file_accuracies(self.normed_h5f_file_list,obj_dump_dir)



#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
class MAIN_DATA_PREP():

    def __init__(self):
        print('Init Class MAIN_DATA_PREP')

    def Do_merge_blocks(self,file_list,stride=[4,4,4],step=[8,8,8]):
        #file_list = glob.glob( os.path.join(GLOBAL_PARA.ETH_A_step_0d5_stride_0d5,   '*_step_0d5_stride_0d5.h5') )
        #file_list = glob.glob( os.path.join(GLOBAL_PARA.ETH_A_stride_1_step_1,   '*_4096.h5') )
        #file_list = glob.glob( os.path.join(GLOBAL_PARA.ETH_A_step_10_stride_10,   '*_blocked.h5_sorted_step_10_stride_10.hdf5') )
        block_step = (np.array(step)).astype(np.int)
        block_stride = (np.array(stride)).astype(np.int)
        #block_stride = (block_step*0.5).astype(np.int)
        print('step = ',block_step)
        print('stride = ',block_stride)

        IsMulti_merge = True
        if not IsMulti_merge:
            for file_name in file_list:
                merged_name = self.merge_blocks_to_new_step(file_name,block_step,block_stride)
                merged_names.append(merged_name)
        else:
            pool = []
            for file_name in file_list:
                p = mp.Process( target=self.merge_blocks_to_new_step, args=(file_name,block_step,block_stride,) )
                p.start()
                pool.append(p)
            for p in pool:
                p.join()

    def merge_blocks_to_new_step(self,base_file_name,larger_step,larger_stride):
        '''merge blocks of sorted raw h5f to get new larger step
        '''
        #new_name = base_file_name.split('_xyz_intensity_rgb')[0] + '_step_' + str(larger_step[0]) + '_stride_' + str(larger_stride[0]) + '.hdf5'
        tmp = rm_file_name_midpart(base_file_name,'_stride_1_step_1')
        new_part = '_stride_' + str(larger_stride[0])+ '_step_' + str(larger_step[0])
        if larger_step[2] != larger_step[0]:
            if larger_step[2]>0:
                new_part += '_z' + str(larger_step[2])
            else:
                new_part += '_zall'

        new_name = os.path.splitext(tmp)[0]  + new_part + '.h5'
        print('new file: ',new_name)
        print('id = ',os.getpid())
        with h5py.File(new_name,'w') as new_h5f:
                base_sh5f = Sorted_H5f(base_h5f,base_file_name)
                new_sh5f = Sorted_H5f(new_h5f,new_name)
                new_sh5f.copy_root_summaryinfo_from_another(self.sorted_h5f,'new_stride')
                new_sh5f.set_step_stride(larger_step,larger_stride)

                read_row_N = 0
                rate_last = -10
                print('%d rows and %d blocks to merge'%(base_sh5f.total_row_N,base_sh5f.total_block_N))
                for dset_name in  self.sorted_h5f:
                    block_i_base = int(dset_name)
                    base_dset_i = self.sorted_h5f[dset_name]
                    block_k_new_ls,i_xyz_new_ls = base_sh5f.get_sub_block_ks(block_i_base,new_sh5f)

                    read_row_N += base_dset_i.shape[0]
                    rate = 100.0 * read_row_N / base_sh5f.total_row_N
                    if int(rate)%10 < 1 and rate-rate_last>5:
                        rate_last = rate
                        print(str(rate),'%   ','  dset_name = ',dset_name, '  new_k= ',block_k_new_ls,'   id= ',os.getpid())
                        new_sh5f.sorted_h5f.flush()

                    for block_k_new in block_k_new_ls:
                        new_sh5f.append_to_dset(block_k_new,base_dset_i)
                    #if rate > 5:
                        #break
                if read_row_N != base_sh5f.total_row_N:
                    print('ERROR!!!  total_row_N = %d, but only read %d'%(base_sh5f.total_row_N,read_row_N))

                total_block_N = 0
                total_row_N = 0
                for total_block_N,dn in enumerate(new_sh5f.sorted_h5f):
                    total_row_N += new_sh5f.sorted_h5f[dn].shape[0]
                total_block_N += 1
                new_sh5f.set_root_attr('total_row_N',total_row_N)
                new_sh5f.set_root_attr('total_block_N',total_block_N)
                print('total_row_N = ',total_row_N)
                print('total_block_N = ',total_block_N)
                new_sh5f.sorted_h5f.flush()

                #new_sh5f.check_xyz_scope()

                if 'sample_merged' in self.actions:
                    Is_gen_obj = 'obj_sampled_merged' in self.actions
                    Is_gen_norm = 'norm_sampled_merged' in self.actions
                    new_sh5f.file_sample(self.sample_num,self.sample_method,\
                                         gen_norm=Is_gen_norm,gen_obj = Is_gen_obj)


    def gen_rawETH_to_h5(self,label_files_glob,line_num_limit=None):
        '''
        transform the data and label to h5 format
        put every dim to a single dataset
            to speed up search and compare of a single dim
        data is large, chunk to speed up slice
        '''

        label_files_list = glob.glob(label_files_glob)
        data_files_list, h5_files_list = self.clean_label_files_list(label_files_list)
        print('%d data-label files detected'%(len(label_files_list)))
        for lf in label_files_list:
            print('\t%s'%(lf))

        for i,label_fn in enumerate(label_files_list):
            data_fn = data_files_list[i]
            h5_fn = h5_files_list[i]
            with open(data_fn,'r') as data_f:
                with open(label_fn,'r') as label_f:
                    with h5py.File(h5_fn,'w') as h5_f:
                        raw_h5f = Raw_H5f(h5_f,h5_fn)
                        raw_h5f.set_num_default_row(GLOBAL_PARA.h5_num_row_1G)
                        data_label_fs = itertools.izip(data_f,label_f)
                        buf_rows = GLOBAL_PARA.h5_num_row_10M*5
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
                                raw_h5f.add_to_dset('xyz',data_buf[:,0:3],start,end)
                                raw_h5f.add_to_dset('intensity',data_buf[:,3:4],start,end)
                                raw_h5f.add_to_dset('color',data_buf[:,4:7],start,end)
                                raw_h5f.add_to_dset('label',label_buf[:,0:1],start,end)
                                h5_f.flush()

                            if line_num_limit != None and k+1 >= line_num_limit:
                                print('break at k= ',k)
                                break

                        self.add_to_dset_all(raw_h5f,data_buf,label_buf,k,buf_rows)
                        raw_h5f.create_done()

                        print('having read %d lines from %s \n'%(k+1,data_fn))
                        #print('h5 file line num = %d'%(xyz_dset.shape[0]))

    def add_to_dset_all(self,raw_h5f,data_buf,label_buf,k,buf_rows):
        k_buf = k%buf_rows
        start = int(k/buf_rows)*buf_rows
        end = k+1
        #print( 'start = %d, end = %d'%(start,end))
        raw_h5f.add_to_dset('xyz',data_buf[0:k_buf+1,0:3],start,end)
        raw_h5f.add_to_dset('intensity',data_buf[0:k_buf+1,3:4],start,end)
        raw_h5f.add_to_dset('color',data_buf[0:k_buf+1,4:7],start,end)
        raw_h5f.add_to_dset('label',label_buf[0:k_buf+1,0:1],start,end)
        raw_h5f.raw_h5f.flush()
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


    def DO_add_geometric_scope_file(self):
        files_glob = os.path.join(self.ETH_training_partBh5_folder,'*.hdf5')
        #files_glob = os.path.join(self.ETH_training_partAh5_folder,'*.hdf5')
        #files_glob = os.path.join(self.Local_training_partAh5_folder,'*.hdf5')
        files_list = glob.glob(files_glob)
        print('%d files detected'%(len(files_list)))

        IsMultiProcess = False
        line_num_limit = 1000*100
        line_num_limit = None

        if not IsMultiProcess:
            for file_name in files_list:
                with h5py.File(file_name,'a') as h5f:
                    raw_h5f = Raw_H5f(h5f,file_name)
                    raw_h5f.add_geometric_scope(line_num_limit)
                #self.add_geometric_scope_file(file_name,line_num_limit)
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


    def main(self,file_list,actions,sample_num=4096,sample_method='random',stride=[4,4,100],step=[8,8,100]):
        # self.actions: [
        # 'merge','sample_merged','obj_sampled_merged','norm_sampled_merged' ]
        self.actions = actions
        self.sample_num = sample_num
        self.sample_method = sample_method
        self.stride = stride
        self.step = step
        if 'merge' in self.actions:
            self.Do_merge_blocks(file_list,self.stride,self.step)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------



def Gen_raw_label_color_obj():
    base_fn = os.path.join(GLOBAL_PARA.ETH_raw_partA,'untermaederbrunnen_station1_xyz_intensity_rgb')
    data_fn = base_fn + '.txt'
    label_fn = base_fn + '.labels'
    obj_fn = base_fn + '.obj'
    obj_labeled_fn = base_fn + '_labeled.obj'
    obj_unlabeled_fn = base_fn + '_unlabeled.obj'
    labeled_N = 0
    unlabeled_N = 0
    with open(data_fn,'r') as data_f:
     with open(label_fn,'r') as label_f:
      with open(obj_fn,'w') as obj_f:
       with open(obj_labeled_fn,'w') as obj_labeled_f:
        with open(obj_unlabeled_fn,'w') as obj_unlabeled_f:
            data_label_fs = itertools.izip(data_f,label_f)
            for k,data_label_line in enumerate(data_label_fs):
                data_k =np.fromstring( data_label_line[0].strip(),dtype=np.float32,sep=' ' )
                label_k = np.fromstring( data_label_line[1].strip(),dtype=np.int16,sep=' ' )
                color_k = Normed_H5f.g_label2color[label_k[0]]
                str_k = 'v ' + ' '.join( [str(d) for d in data_k[0:3] ] ) + ' \t' +\
                    ' '.join( [str(c) for c in color_k] ) + '\n'
                obj_f.write(str_k)
                if label_k == 0:
                    unlabeled_N += 1
                    obj_unlabeled_f.write(str_k)
                else:
                    labeled_N += 1
                    obj_labeled_f.write(str_k)
                if k%(1000*100) == 0:
                    print('gen raw obj %d'%(k))
                if k > 1000*1000*1:
                    break
    total_N = k+1
    print('total_N = %d, labeled_N = %d (%0.3f), unlabeled_N = %d (%0.3f)'%\
          (total_N,labeled_N,1.0*labeled_N/total_N,unlabeled_N,1.0*unlabeled_N/total_N))

def Do_Norm(file_list):
    for fn in file_list:
        with h5py.File(fn,'r') as f:
            sf = Sorted_H5f(f,fn)
            sf.file_normalization()

def Do_sample(file_list):
    #h5f_name = os.path.join(GLOBAL_PARA.ETH_A_stride_8_step_8,\
                      #'bildstein_station5_sub_m80_m5_stride_8_step_8.h5')
    for h5f_name in file_list:
        sample_num =  4096
        sample_method = 'random'
        with h5py.File(h5f_name,'r') as h5f:
            sh5f = Sorted_H5f(h5f,h5f_name)
            sh5f.file_sample(sample_num,sample_method)


def Test_sub_block_ks():
    h5f_name0 = os.path.join(GLOBAL_PARA.ETH_A_stride_1_step_1,'bildstein_station5_stride_1_step_1_sub_m80_m5.h5')
    h5f_name1 = os.path.join(GLOBAL_PARA.ETH_A_stride_1_step_1,'t_bildstein_station5_stride_1_step_1_sub_m80_m5.h5')
    with h5py.File(h5f_name0,'r') as h5f0:
      with h5py.File(h5f_name1,'w') as h5f1:
        sh5f0 = Sorted_H5f(h5f0)
        sh5f1 = Sorted_H5f(h5f1)

        block_step1 = np.array([1,1,1])*4
        block_stride1 = np.array([1,1,1])*2

        sh5f1.copy_root_summaryinfo_from_another(h5f0,'new_stride')
        sh5f1.set_step_stride(block_step1,block_stride1)

        for i,block_k0_str in enumerate( sh5f0.sorted_h5f ):
            if i > 3:
                break
            block_k0 = int(block_k0_str)
            check_flag = True
            #print('block_k0 = ',block_k0)
            block_k1s,i_xyz_1s = sh5f0.get_sub_block_ks(block_k0,sh5f1)
            print('block_k1s = ',len(block_k1s),'   ',block_k1s,'\n')
            for block_k1 in block_k1s:
                # all the space block_k1 should contain block_k0
                block_k0s,i_xyz_0s = sh5f1.get_sub_block_ks(block_k1,sh5f0)
                print('k1 = ',block_k1,'  block_k0 = ',block_k0s,'\nlen = ',len(block_k0s),'\n')
                if block_k0 not in block_k0s:
                    check_flag = False

                for block_k0_ in block_k0s:
                    # all the scope block_k0_ should constain
                    block_k1s_,i_xyz_1s_ = sh5f0.get_sub_block_ks(block_k0_,sh5f1)
                    if block_k1 not in block_k1s_:
                        check_flag = False
            if check_flag:
                print('all check passed')
            else:
                print('check failed')

def Do_Check_xyz():
    #fnl = glob.glob(os.path.join(folder,'*.hdf5'))
    #for fn in fnl:
        raw_fn = os.path.join(GLOBAL_PARA.ETH_A_rawh5,'bildstein_station5_xyz_intensity_rgb.hdf5')
        fn_s = os.path.join( GLOBAL_PARA.ETH_A_stride_1_step_1,'bildstein_station5_sub_m80_m5_stride_2_step_4.h5')
        fn_s = os.path.join( GLOBAL_PARA.ETH_A_stride_1_step_1,'bildstein_station5_sub_m80_m5_stride_4_step_8.h5')
        print('checking equal and  xyz scope of file: ',fn_s)
        with h5py.File(raw_fn,'r') as h5f:
            with h5py.File(fn_s,'r') as sh5f:
                sorted_h5f = Sorted_H5f(sh5f,raw_fn)
                #sorted_h5f.show_summary_info()
               # flag1 = sorted_h5f.check_equal_to_raw(h5f)
               # if flag1:
               #     print('equal check passed')
               # else:
               #     print('equal check failed')
                flag2 = sorted_h5f.check_xyz_scope()
                if flag2:
                    print('xyz scope check passed')
                else:
                    print('xyz scope check failed')


def Do_extract_sub_area():
    folder = GLOBAL_PARA.ETH_A_rawh5
    fnl = glob.glob(os.path.join(folder,'bildstein_station5_stride_1_step_1.h5'))
    #sub_xyz_scope = np.array([[-30,-30,-20],[0,0,50]])
    #new_flag = '_sub_m30_0'
    sub_xyz_scope = np.array([[-80,-80,-20],[-5,-5,50]])
    new_flag = '_sub_m80_m5'
    print('sub_scope:\n',sub_xyz_scope)
    for fn in fnl:
        fn_parts =  os.path.splitext(fn)
        new_name = fn_parts[0]+new_flag+fn_parts[1]
        print('sub file name: ',new_name)
        with h5py.File(fn,'r') as s_h5f:
            sorted_h5f = Sorted_H5f(s_h5f,fn)
            sorted_h5f.extract_sub_area(sub_xyz_scope,new_name)


def Add_sorted_total_row_block_N_onefile(fn):
        print('calculating row_N block_N of: ',fn)
        with h5py.File(fn,'a') as h5f:
            sorted_h5f = Sorted_H5f(h5f,fn)
            rN,bN = sorted_h5f.add_total_row_block_N()
            print('rn= ',rN, '  bN= ',bN,'\n')
def Add_sorted_total_row_block_N():
    folder = GLOBAL_PARA.ETH_A_step_20_stride_10
    fnl = glob.glob(os.path.join(folder,'*.hdf5'))
    IsMulti_aN = False
    if not IsMulti_aN:
        for fn in fnl:
            Add_sorted_total_row_block_N_onefile(fn)
    else:
        p = mp.Pool(3)
        p.map(Add_sorted_total_row_block_N_onefile,fnl)
        p.close()
        p.join()


def Do_gen_raw_obj():
    ETH_training_partAh5_folder =  GLOBAL_PARA.ETH_A_rawh5
    folder_path = ETH_training_partAh5_folder
    file_list = glob.glob( os.path.join(folder_path,'b*.hdf5') )
    IsLabelColor = True
    for fn in file_list:
        print(fn)
        if IsLabelColor:
            meta_fn = '_labeledColor'
        else:
            meta_fn = ''
        obj_fn = os.path.splitext(fn)[0]+meta_fn+'.obj'
        with h5py.File(fn,'r') as  raw_h5_f:
            raw_h5f = Raw_H5f(raw_h5_f)
            raw_h5f.generate_objfile(obj_fn,IsLabelColor)

def Do_gen_sorted_block_obj(file_list):
    for fn in file_list:
        with  h5py.File(fn,'r') as h5f:
            sorted_h5f = Sorted_H5f(h5f,fn)
            sorted_h5f.gen_file_obj(True)

def Do_gen_normed_obj(file_list):
    for fn in file_list:
        with  h5py.File(fn,'r') as h5f:
            norm_h5f = Normed_H5f(h5f,fn)
            norm_h5f.gen_gt_pred_obj()

def Do_gen_gt_pred_objs(file_list=None):
    if file_list == None:
        folder = GLOBAL_PARA.stanford_indoor3d_globalnormedh5_stride_0d5_step_1_4096
        # many chairs and tables
        #file_list = glob.glob(os.path.join(folder,'Area_1_office_16_stride_0.5_step_1_random_4096_globalnorm.nh5'))
        # simple only one table
        file_list = glob.glob(os.path.join(folder,'Area_6_pantry_1_stride_0.5_step_1_random_4096_globalnorm.nh5'))
    for fn in file_list:
        with h5py.File(fn,'r') as h5f:
            norm_h5f = Normed_H5f(h5f,fn)
            norm_h5f.gen_gt_pred_obj_examples()

def gen_file_list(folder,format='h5'):
    file_list = glob.glob( os.path.join(folder,'*.'+format) )
    print(file_list)
    with open(os.path.join(folder,'all_files.txt'),'w') as f:
        for fn in file_list:
            base_filename = os.path.basename(fn)
            base_dirname = os.path.basename( os.path.dirname(fn) )
            base_dir_file_name = os.path.join(base_dirname,base_filename)
            f.write( base_dir_file_name )
            print(base_dir_file_name)
    print('all file list file write OK ')



netpro_test='/short/dh01/yx2146/pointnet/data/net_provider_test'
class Indoor3d_Process():
    '''
    source:  from "http://buildingparser.stanford.edu/dataset.html"
    Tihe work flow of processing stanford_indoor3d data:
        (1) collectto each room to format: [x y z r g b label]. Done by Qi's code
        (2) gen each room to Raw_H5f
        (3) sort each room to Sorted_H5f with step = stride = 0.5
        (4) merge each room to Sorted_H5f with step = 1 & stride = 0.5
        (5) sample each block to NUM_POINT points and normalize each block
    '''
    @staticmethod
    def gen_stanford_indoor3d_to_rawh5f():
        file_list = glob.glob( os.path.join( GLOBAL_PARA.stanford_indoor3d_collected_path,\
                                            '*.npy' ) )
        file_list = glob.glob(os.path.join(netpro_test,'*.npy'))
        for fn in file_list:
            h5_fn = os.path.splitext(fn)[0]+'.h5'
            with h5py.File(h5_fn,'w') as h5f:
                raw_h5f = Raw_H5f(h5f,h5_fn)
                data = np.load(fn)
                num_row = data.shape[0]
                raw_h5f.set_num_default_row(num_row)
                raw_h5f.append_to_dset('xyz',data[:,0:3])
                raw_h5f.append_to_dset('color',data[:,3:6])
                raw_h5f.append_to_dset('label',data[:,6:7])
                raw_h5f.create_done()

    @staticmethod
    def Do_SortSpace():
        file_list = glob.glob( os.path.join(GLOBAL_PARA.stanford_indoor3d_rawh5, \
                    '*.h5') )
        file_list = glob.glob(os.path.join(netpro_test,'Area_6_OFFICE_1.h5'))
        block_step_xyz = [0.5,0.5,0.5]
        SortSpace(file_list,block_step_xyz)
    @staticmethod
    def MergeSampleNorm():
        file_list = glob.glob( os.path.join(GLOBAL_PARA.stanford_indoor3d_stride_0d5_step_0d5, \
                    'Area_6_office_1*_stride_0.5_step_0.5.sh5') )
        #file_list = glob.glob(os.path.join(netpro_test,'Area_6_OFFICE_1_stride_0.5_step_0.5.sh5'))
        new_stride = [1,1,-1]
        new_step = [1,1,-1]
        more_actions_config = {}
        more_actions_config['actions'] = ['merge','sample_merged','norm_sampled_merged']
        #more_actions_config['actions'] = ['merge','obj_merged','sample_merged','obj_sampled_merged']
        more_actions_config['sample_num'] = 4096
        more_actions_config['sample_method'] = 'random'
        for fn in file_list:
            with h5py.File(fn,'r') as f:
                sorted_h5f = Sorted_H5f(f,fn)
                sorted_h5f.merge_to_new_step(new_stride,new_step,more_actions_config)

    @staticmethod
    def Norm():
        file_list = glob.glob( os.path.join(GLOBAL_PARA.stanford_indoor3d_stride_0d5_step_1_4096,\
                              '*.sh5') )
        for fn in file_list:
            with h5py.File(fn,'r') as f:
                sorted_h5f = Sorted_H5f(f,fn)
                sorted_h5f.file_normalization()

    @staticmethod
    def MergeAreaRooms():
        # and add area num dataset
        for area_no in range(1,7):
            area_str = 'Area_'+str(area_no)
            path = GLOBAL_PARA.stanford_indoor3d_normed_stride_0d5_step_1_4096
            file_list = glob.glob( os.path.join(path, \
                                    area_str+'*.nh5' ) )
            print('file num = %d'%(len(file_list)))
            postfix = os.path.basename(file_list[0]).split('_stride_')[1]
            root_path = os.path.dirname(path)
            merged_fn = os.path.join(root_path,area_str+'_stride_'+postfix)
            print('merged file name: %s'%(merged_fn))
            with h5py.File(merged_fn,'w') as f:
                merged_normed_h5f = Normed_H5f(f,merged_fn)
                for k,fn in enumerate(file_list):
                    if k==0:
                        with h5py.File(fn,'r') as f0:
                            normed_h5f_0 = Normed_H5f(f0,fn)
                            data_shape = normed_h5f_0.get_data_shape()
                            merged_normed_h5f.create_dsets(0,data_shape[1],data_shape[2])
                            merged_normed_h5f.create_areano_dset(0,data_shape[1])
                    merged_normed_h5f.merge_file(fn)



def main(file_list):

    outdoor_prep = MAIN_DATA_PREP()
    actions = ['merge','sample_merged','obj_sampled_merged','norm_sampled_merged']
    actions = ['merge','sample_merged','norm_sampled_merged']
    outdoor_prep.main(file_list,actions,sample_num=4096,sample_method='random',\
                      stride=[8,8,-1],step=[8,8,-1])

    #outdoor_prep.Do_sort_to_blocks()
    #Do_extract_sub_area()
    #outdoor_prep.test_sub_block_ks()
    #outdoor_prep.DO_add_geometric_scope_file()
    #outdoor_prep.DO_gen_rawETH_to_h5()

if __name__ == '__main__':
 #   file_list = glob.glob( os.path.join(GLOBAL_PARA.ETH_A_stride_1_step_1, \
 #               '*_m5.h5') )
    #file_list = glob.glob( os.path.join(GLOBAL_PARA.ETH_A_stride_8_step_8, \
                #'*_4096.h5') )
   # file_list = glob.glob( os.path.join(GLOBAL_PARA.seg_train_path, \
   #             '*.h5') )
    #main(file_list)
    #Do_gen_raw_obj()
    #Add_sorted_total_row_block_N()
    #Do_Check_xyz()
    #Test_sub_block_ks()
    #Do_sample()
    #Do_gen_sorted_block_obj(file_list)
    #Do_gen_normed_obj(file_list)
    #Do_Norm(file_list)
    #gen_file_list(GLOBAL_PARA.seg_train_path)
    #Do_gen_gt_pred_objs()
    Write_all_file_accuracies()
    #Normed_H5f.show_all_colors()
    #Gen_raw_label_color_obj()
    #Indoor3d_Process.gen_stanford_indoor3d_to_rawh5f()
    #Indoor3d_Process.Do_SortSpace()
    #Indoor3d_Process.MergeSampleNorm()
    #Indoor3d_Process.Norm()
    T = time.time() - START_T
    print('exit main, T = ',T)
