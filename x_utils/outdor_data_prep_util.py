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

g_h5_chunk_row_step_1M = 50*1000


class GLOBAL_PARA():
    ETH_traing_A =  '/short/dh01/yx2146/Dataset/ETH_Semantic3D_Dataset/training'
    ETH_A_rawh5 = os.path.join( ETH_traing_A,'part_A_rawh5' )
    ETH_A_stride_1_step_1 = os.path.join( ETH_traing_A, 'A_stride_1_step_1' )
    ETH_A_stride_2_step_2 = os.path.join( ETH_traing_A, 'A_stride_2_step_2' )
    ETH_A_stride_4_step_8 = os.path.join( ETH_traing_A, 'A_stride_4_step_8' )
    ETH_A_stride_5_step_5 = os.path.join( ETH_traing_A, 'A_stride_5_step_5' )
    ETH_A_stride_8_step_8 = os.path.join( ETH_traing_A, 'A_stride_8_step_8' )
    ETH_A_stride_20_step_10 = os.path.join( ETH_traing_A, 'A_stride_20_step_10' )
    h5_chunk_row_step_1M = g_h5_chunk_row_step_1M
    h5_chunk_row_step_10M = h5_chunk_row_step_1M * 10
    h5_chunk_row_step_100M = h5_chunk_row_step_1M * 100
    h5_chunk_row_step_1G = h5_chunk_row_step_1M * 1000
    h5_chunk_row_step =  h5_chunk_row_step_10M




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
    h5_chunk_row_step_1M = g_h5_chunk_row_step_1M
    def __init__(self,raw_h5_f,file_name='Unknown'):
        self.raw_h5f = raw_h5_f
        self.get_summary_info(raw_h5_f)
        self.file_name = file_name

    def get_summary_info(self,raw_h5_f):
        self.xyz_dset = raw_h5_f['xyz']
        self.label_dset = raw_h5_f['label']
        self.color_dset = raw_h5_f['color']
        self.intensity_dset = raw_h5_f['intensity']

        self.total_row_N = self.xyz_dset.shape[0]
        self.xyz_max = self.xyz_dset.attrs['max']
        self.xyz_min = self.xyz_dset.attrs['min']
        self.xyz_scope = self.xyz_max - self.xyz_min

    def generate_objfile(self,obj_file_name):
        with open(obj_file_name,'w') as out_obj_file:
            xyz_dset = self.xyz_dset
            color_dset = self.color_dset

            row_step = self.h5_chunk_row_step_1M * 10
            row_N = xyz_dset.shape[0]
            for k in range(0,row_N,row_step):
                end = min(k+row_step,row_N)
                xyz_buf_k = xyz_dset[k:end,:]
                color_buf_k = color_dset[k:end,:]
                buf_k = np.hstack((xyz_buf_k,color_buf_k))
                for j in range(0,buf_k.shape[0]):
                    str_j = 'v   ' + '\t'.join( ['%0.5f'%(d) for d in  buf_k[j,0:3]]) + '  \t'\
                    + '\t'.join( ['%d'%(d) for d in  buf_k[j,3:6]]) + '\n'
                    out_obj_file.write(str_j)

                rate = int(100.0 * end / row_N)
                e = row_step / row_N
                if rate > 5 and rate % 10 <= e:
                    print('gen raw obj: %d%%'%(rate))

    def add_geometric_scope(self,line_num_limit=None):
        ''' calculate the geometric scope of raw h5 data, and add the result to attrs of dset'''
        begin = time.time()
        max_xyz = -np.ones((3))*1e10
        min_xyz = np.ones((3))*1e10

        xyz_dset = self.xyz_dset
        row_step = self.h5_chunk_row_step_1M
        print('There are %d lines in xyz dataset of file: %s'%(xyz_dset.shape[0],self.file_name))
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
        xyz_dset.attrs['max'] = max_xyz
        xyz_dset.attrs['min'] = min_xyz
        max_str = '  '.join([ str(e) for e in max_xyz ])
        min_str = '  '.join([ str(e) for e in min_xyz ])
        print('File: %s\n\tmax_str=%s\n\tmin_str=%s'%(self.file_name,max_str,min_str) )
        print('T=',time.time()-begin)


class Sorted_H5f():
    stride_to_align = 8
    h5_chunk_row_step_1M = g_h5_chunk_row_step_1M
    def __init__(self,sorted_h5f,file_name=None):
        self.sorted_h5f = sorted_h5f
        self.get_summary_info()
        if file_name != None:
            self.file_name = file_name
        else:
            self.file_name = 'unknown'

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
        print('add_total_row_block_N:  file: %s \n   total_row_N = %d,  total_block_N = %d'%(self.file_name,self.total_row_N,self.total_block_N))
        return total_row_N, n+1

    def copy_root_summaryinfo_from_another(self,h5f0,copy_flag):
        if copy_flag =='new_stride':
            attrs = ['xyz_max','xyz_min']
        elif copy_flag == 1:
            attrs = ['xyz_max','xyz_min','total_row_N','total_block_N']
        elif copy_flag == 'sub':
            attrs = ['xyz_max','xyz_min','block_step','block_stride']
        elif copy_flag == 'sampled':
            attrs = ['xyz_max','xyz_min','total_block_N','block_step','block_stride']
        elif copy_flag == 3:
            attrs = ['xyz_max','xyz_min','total_row_N','total_block_N','block_step','block_stride']
        else:
            attrs = ['xyz_max','xyz_min']

        for attr in attrs:
            if attr in h5f0.attrs:
                self.sorted_h5f.attrs[attr] = h5f0.attrs[attr]
        self.get_summary_info()
    def copy_root_geoinfo_from_raw(self,raw_h5f,attrs=['xyz_max','xyz_min']):
        for attr in attrs:
            if hasattr(raw_h5f,attr):
                self.sorted_h5f.attrs[attr] = getattr(raw_h5f,attr)
        self.get_summary_info()

    def set_step_stride(self,block_step,block_stride):
        xyz_min = self.sorted_h5f.attrs['xyz_min']
        xyz_max = self.sorted_h5f.attrs['xyz_max']
        xyz_min_aligned = xyz_min - xyz_min % self.stride_to_align - [0,0,1]
        xyz_max_aligned = xyz_max - xyz_max % 1 + 1
        xyz_scope_aligned =  xyz_max_aligned - xyz_min_aligned
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
                            import pdb; pdb.set_trace()  # XXX BREAKPOINT

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
                            import pdb; pdb.set_trace()  # XXX BREAKPOINT

                        else:
                            #print('both min and max check passed, i_xyz= ',[i_x,i_y,i_z])
                            i_xyz_new_list.append(i_xyz_new)
                            block_k_new_list.append(block_k_new)
        return block_k_new_list,i_xyz_new_list

    def get_blocked_dset(self,block_k,new_set_default_rows=None):
        if not type(block_k) is int:
            block_k = int(block_k)

        dset_name = str(block_k)
        if dset_name in self.sorted_h5f:
            return self.sorted_h5f[dset_name]
        if new_set_default_rows==None:
            new_set_default_rows = self.h5_chunk_row_step_1M
        n = 9
        #dset = self.h5f_blocked.create_dataset( dset_name,shape=(new_set_default_rows,n),\
                #maxshape=(None,n),dtype=np.float32,chunks=(self.h5_chunk_row_step_1M/5,n) )
        dset = self.sorted_h5f.create_dataset( dset_name,shape=(new_set_default_rows,n),\
                maxshape=(None,n),dtype=np.float32,compression="gzip"  )
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

    def check_equal_to_raw(self,raw_f):
        raw_xyz_set = raw_f['xyz']
        raw_color_set = raw_f['color']
        raw_label_set = raw_f['label']
        raw_intensity_set = raw_f['intensity']
        check_flag = True
        for k,block_k in enumerate(self.sorted_h5f):
            #print('checing block %s'%(block_k))
            dset_k = self.sorted_h5f[block_k]
            step = max(int(dset_k.shape[0]/30),1)
            flag_k = True
            for i in range(0,dset_k.shape[0],step):
                sorted_d_i = dset_k[i,0:8]
                raw_k = dset_k[i,8]
                if raw_k < 0 or raw_k > 16777215: # for float32, it is not accurate again
                    continue
                raw_d_i = np.concatenate(  [raw_xyz_set[raw_k,:],raw_color_set[raw_k,:],raw_label_set[raw_k,:],raw_intensity_set[raw_k,:]] )
                error = raw_d_i - sorted_d_i
                err = np.linalg.norm( error )
                if err != 0:
                    flag_k = False
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

    def append_to_dset(self,aim_block_k,source_dset,vacant_size=0,sample_choice=None):
        '''
        if append frequently to one dataset, vacant_size > 0 to avoid frequent resize
        '''
        if sample_choice == None:
            sample_choice = np.arange(source_dset.shape[0])
        aim_dset = self.get_blocked_dset(aim_block_k,vacant_size)
        row_step = self.h5_chunk_row_step_1M * 10
        org_row_N = aim_dset.attrs['valid_num']
        new_row_N = sample_choice.size
        aim_dset.resize((org_row_N+new_row_N+vacant_size,aim_dset.shape[1]))
        for k in range(0,new_row_N,row_step):
            end = min(k+row_step,new_row_N)
            choice_k = sample_choice[k:end]
            aim_dset[org_row_N+k:org_row_N+end,:] = source_dset[choice_k,:]
            aim_dset.attrs['valid_num'] = end + org_row_N

    def generate_one_block_to_object(self,block_k,out_obj_file):
        row_step = self.h5_chunk_row_step_1M * 10
        dset_k = self.get_blocked_dset(block_k)
        row_N = dset_k.shape[0]
        for k in range(0,row_N,row_step):
            end = min(k+row_step,row_N)
            buf_k = dset_k[k:end,0:6]
            #buf_k[:,0:3] -= middle
            for j in range(0,buf_k.shape[0]):
                str_j = 'v ' + ' '.join( ['%0.3f'%(d) for d in  buf_k[j,0:3]]) + ' \t'\
                 + ' '.join( ['%d'%(d) for d in  buf_k[j,3:6]]) + '\n'
                out_obj_file.write(str_j)

    def generate_blocks_to_object(self,obj_folder):
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
                out_fn = os.path.join(obj_folder,dset_name+'_'+str(row_N)+'.obj')
                with open(out_fn,'w') as out_f:
                    self.generate_one_block_to_object(dset_name,out_f)
                n += row_N
                rate = 100.0 * n / self.total_row_N
                if int(rate) % 2 == 0 and rate - last_rate > 3:
                    last_rate = rate
                    print('%0.2f%% generating file: %s'%(rate,out_fn) )

                info_str = 'dset: %s \tN= %d   \tmin=%s   \tmax=%s \n'%(dset_name,self.sorted_h5f[dset_name].shape[0], np.array_str(min_i), np.array_str(max_i)  )
                info_f.write(info_str)
                print(info_str)
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

    def sample_k(self,k,aim_N):
        dset = self.sorted_h5f[str(k)]
        N = dset.shape[0]
        if N == aim_N:
            return
        sample = np.random.choice(N,aim_N)
        new_data = dset[sample,...]

    def file_sample(self,sample_num,sampled_filename):
        with h5py.File(sampled_filename,'w') as sampled_h5f:
            sampled_sh5f = Sorted_H5f(sampled_h5f,sampled_filename)
            sampled_sh5f.copy_root_summaryinfo_from_another(self.sorted_h5f,'sampled')
            for k_str in self.sorted_h5f:
                dset_k = self.sorted_h5f[k_str]
                sample_choice = sample(dset_k.shape[0],sample_num)
                sampled_sh5f.append_to_dset(int(k_str),dset_k,vacant_size=0,sample_choice=sample_choice)


    def normalization(self):
        '''
        (1) xyz/max
        (2) xy-min-block_size/2  (only xy)
        '''
        pass

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


class OUTDOOR_DATA_PREP():

    def __init__(self):
        print('Init Class OUTDOOR_DATA_PREP')
        #print(self.ETH_training_partAh5_folder)


    def Do_merge_blocks(self,stride_x=2,step_x=4):
        #file_list = glob.glob( os.path.join(GLOBAL_PARA.ETH_A_step_0d5_stride_0d5,   '*_step_0d5_stride_0d5.h5') )
        file_list = glob.glob( os.path.join(GLOBAL_PARA.ETH_A_stride_1_step_1,   'bildstein_station5_stride_1_step_1*.h5') )
        #file_list = glob.glob( os.path.join(GLOBAL_PARA.ETH_A_step_10_stride_10,   '*_blocked.h5_sorted_step_10_stride_10.hdf5') )
        block_step = (np.array([1,1,1])*step_x).astype(np.int)
        block_stride = (np.array([1,1,1])*stride_x).astype(np.int)
        #block_stride = (block_step*0.5).astype(np.int)
        print('step = ',block_step)
        print('stride = ',block_stride)

        IsMulti_merge = True
        if not IsMulti_merge:
            for file_name in file_list:
                self.merge_blocks_to_new_step(file_name,block_step,block_stride)
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
        new_name = os.path.splitext(tmp)[0]  + '_stride_' + str(larger_stride[0])+ '_step_' + str(larger_step[0]) + '.h5'
        print('new file: ',new_name)
        print('id = ',os.getpid())
        with  h5py.File(base_file_name,'r') as base_h5f:
            with h5py.File(new_name,'w') as new_h5f:
                base_sh5f = Sorted_H5f(base_h5f,base_file_name)
                new_sh5f = Sorted_H5f(new_h5f,new_name)
                new_sh5f.copy_root_summaryinfo_from_another(base_h5f,'new_stride')
                new_sh5f.set_step_stride(larger_step,larger_stride)

                read_row_N = 0
                rate_last = -10
                print('%d rows and %d blocks to merge'%(base_sh5f.total_row_N,base_sh5f.total_block_N))
                for dset_name in  base_h5f:
                    block_i_base = int(dset_name)
                    base_dset_i = base_h5f[dset_name]
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

                new_sh5f.check_xyz_scope()


    def gen_rawETH_to_h5(self,label_files_glob,line_num_limit=None):
        '''
        transform the data and label to h5 format
        put every dim to a single dataset
            to speed up search and compare of a single dim
        data is large, chunk to speed up slice
        '''
        h5_chunk_row_step_1M = g_h5_chunk_row_step_1M
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
            with open(data_fn,'r') as data_f:
                with open(label_fn,'r') as label_f:
                    with h5py.File(h5_fn,'w') as h5_f:
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

    def sort_to_blocks(self,file_name,block_step_x=1):
        '''
        split th ewhole scene to space sorted small blocks
        The whole scene is a group. Each block is one dataset in the group.
        The block attrs represents the field.
        '''
        print(file_name)
        block_step = np.ones((3))*block_step_x
        print('block step = ',block_step)
        self.row_num_limit = None

        tmp = block_step[0]
        if tmp % 1 ==0:
            tmp = int(tmp)
        fn = rm_file_name_midpart(file_name,'_intensity_rgb')
        fn = rm_file_name_midpart(fn,'_xyz')
        blocked_file_name = os.path.splitext(fn)[0]+'_stride_'+str(tmp)+'_step_'+str(tmp)+'.h5'
        with h5py.File(blocked_file_name,'w') as h5f_blocked:
            with h5py.File(file_name,'r') as h5_f:
                self.raw_h5f = Raw_H5f(h5_f,file_name)
                self.s_h5f = Sorted_H5f(h5f_blocked,blocked_file_name)
                self.s_h5f.set_root_attr('elements','xyz-color-label-intensity-raw_k')
                self.s_h5f.copy_root_geoinfo_from_raw( self.raw_h5f )
                self.s_h5f.set_step_stride(block_step,block_step)

                #self.row_num_limit = int(self.raw_h5f.total_row_N/1000)

                row_step = GLOBAL_PARA.h5_chunk_row_step_1M*8
                sorted_buf_dic = {}
                raw_row_N = self.raw_h5f.xyz_dset.shape[0]
                for k in range(0,raw_row_N,row_step):
                    end = min(k+row_step,raw_row_N)
                    raw_buf = np.zeros((end-k,9))
                    #t0_k = time.time()
                    #print('start read %d:%d'%(k,end))
                    raw_buf[:,0:3] = self.raw_h5f.xyz_dset[k:end,:]
                    raw_buf[:,3:6] = self.raw_h5f.color_dset[k:end,:]
                    raw_buf[:,6:7] = self.raw_h5f.label_dset[k:end,:]
                    raw_buf[:,7:8] = self.raw_h5f.intensity_dset[k:end,:]
                    if end < 16777215: # this is the largest int float32 can acurately present
                        raw_buf[:,8] = np.arange(k,end)
                    else:
                        raw_buf[:,8] = -1
                    #t1_k = time.time()
                    #print('all read T=',time.time()-read_t0)

                    #t2_0_k = time.time()

                    sorted_buf_dic={}
                    self.sort_buf(raw_buf,k,sorted_buf_dic)

                    #t2_1_k = time.time()
                    self.h5_write_buf(sorted_buf_dic)

                    #t2_2_k = time.time()
                    if int(k/row_step) % 1 == 0:
                        print('%%%.1f  line[ %d:%d ] block_N = %d'%(100.0*end/self.raw_h5f.total_row_N, k,end,len(sorted_buf_dic)))
                         #print('line: [%d,%d] blocked   block_T=%f s, read_T=%f ms, cal_t = %f ms, write_t= %f ms'%\
                               #(k,end,time.time()-t0_k,(t1_k-t0_k)*1000,(t2_1_k-t2_0_k)*1000, (t2_2_k-t2_1_k)*1000 ))
                    if hasattr(self,'row_num_limit') and self.row_num_limit!=None and  end>=self.row_num_limit:
                    #if k /row_step >3:
                        print('break read at k= ',end)
                        break

#                total_block_N = 0
#                total_row_N = 0
#                for dset_name_i in self.h5f_blocked:
#                    total_block_N += 1
#                    total_row_N += self.h5f_blocked[dset_name_i].shape[0]
#                self.h5f_blocked.attrs['total_block_N'] = total_block_N
#                self.h5f_blocked.attrs['total_row_N'] = total_row_N

                total_row_N,total_block_N = self.s_h5f.add_total_row_block_N()

                if total_row_N != self.raw_h5f.total_row_N:
                    print('ERROR: blocked total_row_N= %d, raw = %d'%(total_row_N,self.raw_h5f.total_row_N))
                print('total_block_N = ',total_block_N)

                check = self.s_h5f.check_equal_to_raw(h5_f) & self.s_h5f.check_xyz_scope()
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
            self.s_h5f.append_to_dset(block_k,sorted_buf_dic[block_k],vacant_size=GLOBAL_PARA.h5_chunk_row_step_1M)

#            dset_k =  self.s_h5f.get_blocked_dset(block_k)
#            valid_n = dset_k.attrs['valid_num']
#            new_valid_n = valid_n + sorted_buf_dic[block_k].shape[0]
#            while dset_k.shape[0] < new_valid_n:
#                dset_k.resize(( dset_k.shape[0]+self.h5_chunk_row_step_1M,dset_k.shape[1]))
#            dset_k[valid_n:new_valid_n,:] = sorted_buf_dic[block_k]
#            dset_k.attrs['valid_num'] = new_valid_n
        self.s_h5f.rm_invalid_data()
        self.s_h5f.sorted_h5f.flush()
    #-------------------------------------------------------------------------------
    '''
    Implement Functions
    '''
    #-------------------------------------------------------------------------------

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

    def Do_sort_to_blocks(self):
        ETH_raw_h5_glob =glob.glob(  os.path.join( GLOBAL_PARA.ETH_A_rawh5,'b*.hdf5') )
        #ETH_raw_h5_glob =glob.glob(  os.path.join( '/short/dh01/yx2146/Dataset/ETH_Semantic3D_Dataset/training/tmp_test/bildstein_station5_xyz_intensity_rgb.hdf5') )

        block_step_x = 1
        IsMulti = False
        if not IsMulti:
            for fn in ETH_raw_h5_glob:
                print('sort file: ',fn)
                self.sort_to_blocks(fn,block_step_x)
        else:
            #pool = mp.Pool( max(mp.cpu_count()/2,1) )
            print('cpu_count= ',mp.cpu_count())
            pool = mp.Pool()
            for fn in ETH_raw_h5_glob:
                pool.apply_async(self.sort_to_blocks(fn,block_step_x))
            pool.close()
            pool.join()

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------



def Do_gen_raw_obj():
    ETH_training_partAh5_folder = '/short/dh01/yx2146/Dataset/ETH_Semantic3D_Dataset/training/part_A_rawh5'
    folder_path = ETH_training_partAh5_folder
    file_list = glob.glob( os.path.join(folder_path,'sg27_station5*.hdf5') )
    for fn in file_list:
        print(fn)
        obj_fn = os.path.splitext(fn)[0]+'.obj'
        with h5py.File(fn,'r') as  raw_h5_f:
            raw_h5f = Raw_H5f(raw_h5_f)
            raw_h5f.generate_objfile(obj_fn)

def Do_gen_sorted_block_obj():
    folder_path = GLOBAL_PARA.ETH_A_stride_8_step_8
    file_list = glob.glob( os.path.join(folder_path,\
                 'bildstein_station5_sub_m80_m5_stride_8_step_8.h5') )
    for fn in file_list:
        base_fn = os.path.basename(fn)
        base_fn = os.path.splitext(base_fn)[0]
        obj_folder = os.path.join(folder_path,base_fn)
        if not os.path.exists(obj_folder):
            os.makedirs(obj_folder)
        with  h5py.File(fn,'r') as h5f:
            sorted_h5f = Sorted_H5f(h5f)
            sorted_h5f.generate_blocks_to_object(obj_folder)


def main():
    outdoor_prep = OUTDOOR_DATA_PREP()
    #outdoor_prep.Do_sort_to_blocks()
    #Do_extract_sub_area()
    outdoor_prep.Do_merge_blocks(stride_x=8,step_x=8)
    #outdoor_prep.Do_merge_blocks(stride_x=4,step_x=8)
    #outdoor_prep.Do_merge_blocks(stride_x=2,step_x=4)
    #outdoor_prep.Do_merge_blocks(stride_x=1,step_x=2)
    #outdoor_prep.test_sub_block_ks()
    #outdoor_prep.DO_add_geometric_scope_file()
    #outdoor_prep.DO_gen_rawETH_to_h5()

if __name__ == '__main__':
    #main()
    Do_gen_sorted_block_obj()
    #Do_gen_raw_obj()
    #Add_sorted_total_row_block_N()
    #Do_Check_xyz()
    #Test_sub_block_ks()
    T = time.time() - START_T
    print('exit main, T = ',T)
