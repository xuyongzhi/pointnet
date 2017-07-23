from  __future__ import print_function
import numpy as np
import os
import sys
import glob
import itertools
import multiprocessing as mp
import time
import argparse

BASE_DIR = os.path.dirname( os.path.abspath(__file__) )
ROOT_DIR = os.path.dirname( BASE_DIR )
sys.path.append(BASE_DIR)



def remove_column_from_str(line_str,idx_to_rm):
    data_list = line_str.split()
    data_list.pop(idx_to_rm)
    new_line_str = '\t'.join(str(d) for d in data_list)
    return new_line_str

def get_part_file_name(file_name,idx):
    base_name,format_name = os.path.splitext(file_name)
    base_name += '-part-'+str(idx)
    return base_name+format_name

def collect_file_slice(data_f_slice,label_f_slice,out_f_slice,line_num_limit_slice):
    '''
    args: the sliced / normal file object
    '''
    for i,e in enumerate(zip(data_f_slice,label_f_slice)):
        data_line = e[0]
        data_line = remove_column_from_str(data_line.strip(),3)
        labels_line = e[1]
        out_line = data_line + '\t' + labels_line
        out_f_slice.write(out_line)

        if line_num_limit_slice!=None and i > line_num_limit_slice:
            break
        if i%500000 == 0:
            print('line %d in %s'%(i))


def collect_one_file_multiprocess(total_line_nums,data_file_name,
                                  label_file_name,out_file_name,line_num_limit,num_workers):

    if line_num_limit == None:
        step = int( total_line_nums / num_workers ) + 1
    else:
        step = int( line_num_limit / num_workers ) + 1
    data_files = []
    label_files = []
    out_files = []
    for i in range(num_workers):
        out_file_name_i = get_part_file_name(out_file_name)
        start = i*step
        end = (i+1)*step
        if i==num_workers-1:
            end = line_num_limit
            with open(data_file_name,'r') as df, open(label_file_name,'r') as lf,open(out_file_name_i,'w') as of:
                data_files[i] = itertools.islice( df,start,end )
                label_files[i] = itertools.islice( lf,start,end )
                out_files[i] = itertools.islice( of,start,end )
                collect_file_slice(data_files[i],label_files[i],out_files[i],None)


def collect_one_file_singleprocess(data_file_name,label_file_name,out_file_name,line_num_limit):

    print('start collecting ',data_file_name )
    with open(data_file_name,'r') as data_f, open(label_file_name,'r') as label_f, open(out_file_name,'w')  as out_f:
        collect_file_slice(data_f,label_f,out_f)
#        for i,e in enumerate(zip(data_f,label_f)):
#            data_line = e[0]
#            data_line = remove_column_from_str(data_line.strip(),3)
#            labels_line = e[1]
#            out_line = data_line + '\t' + labels_line
#            out_f.write(out_line)
#
#            if line_num_limit!=None and i > line_num_limit:
#                break
#            if i%500000 == 0:
#                print('line %d in %s'%(i,os.path.basename(data_file_name)))
    print('end writing %s'%(out_file_name) )

def get_total_line_num(file_name):
    with open(file_name,'r') as f:
        N = 0
        for __ in f:
            N += 1
        return N

def collect_one_file(data_file_name,label_file_name,out_file_name,line_num_limit,OneFile_WorkersNum):
    if OneFile_WorkersNum > 0:
        total_line_nums = 500*10000
        collect_one_file_multiprocess(total_line_nums,data_file_name,
                                      label_file_name,out_file_name,line_num_limit,OneFile_WorkersNum)
    else:
        collect_one_file_singleprocess(data_file_name,label_file_name,out_file_name,line_num_limit)



def collect_ETH3d(ETH_DataFolder,LabeledData_OutFolder):
    import shutil

    # Important parameters
    Files_WorkersNum = 0
    OneFile_WorkersNum = 0
    line_num_limit = 1000

    if not os.path.exists(ETH_DataFolder):
        print('ERROR   ETH_DataFolder not exists: ',ETH_DataFolder)
        return
    labels_file_list = glob.glob(ETH_DataFolder+'/*.labels')

    if os.path.exists(LabeledData_OutFolder):
        shutil.rmtree(LabeledData_OutFolder)
    os.makedirs(LabeledData_OutFolder)

    print(len(labels_file_list),' files detected')
    if Files_WorkersNum > 0:
        p = mp.Pool(mp.cpu_count()-0)

    for label_file_name in labels_file_list:
        label_file_basename = os.path.splitext(label_file_name)[0]
        data_file_name = label_file_basename+'.txt'
        if  not os.path.isfile(data_file_name):
            print('file not exist: ',data_file_name)
            continue
        out_file_name = os.path.basename(label_file_basename) + '_labeled.txt'
        out_file_name = os.path.join(LabeledData_OutFolder,out_file_name)

        if Files_WorkersNum > 0:
            p.apply_async(collect_one_file,
                          args=(data_file_name,label_file_name,out_file_name,line_num_limit,))
        else:
            collect_one_file(data_file_name,label_file_name,out_file_name,line_num_limit,OneFile_WorkersNum)
    if Files_WorkersNum > 0:
        p.close()
        p.join()
        print('multi process')
    else:
        print('single process')

def data_label_files_split(label_file_name,splited_folder=None,split_N=5):
    line_num = get_total_line_num(label_file_name)
    label_file_basename = os.path.splitext(label_file_name)[0]
    data_file_name = label_file_basename+'.txt'
    if not os.path.exists(data_file_name) and os.path.exists(label_file_name):
        print('ERROR file not exist')
        return
    if splited_folder == None:
        splited_folder = os.path.splitext(label_file_name)[0]
    if not os.path.exists(splited_folder):
        os.makedirs(splited_folder)

    file_split(label_file_name,splited_folder,split_N,line_num)
    file_split(data_file_name,splited_folder,split_N,line_num)

def file_split(in_file_name,splited_folder,split_N,total_line_num):
    ''' split file in_file_name to  split_N slices, store in splited_folder '''
    file_basename = os.path.splitext( os.path.basename(in_file_name) )[0]
    out_names = [ os.path.join( splited_folder,file_basename+'-slice-'+str(i) +'.txt')  for i in range(split_N) ]
    with open(in_file_name,'r') as f:
        step = total_line_num / split_N + 1
        out_f = []
        for k in range(split_N):
            out_f_k = open(out_names[k],'w')
            out_f.append(out_f_k)
        j = -1
        for i,line  in enumerate(f):
            if i%step == 0 and j < split_N-1:
                j += 1
            out_f[j].write(line)

        for k in range(split_N):
            out_f[k].close()

def merge_spilted_files(splited_folder,merged_folder=None):
    if merged_folder == None:
        merged_folder = os.path.join(splited_folder,'merged')
    if not os.path.exists(merged_folder):
        os.makedirs(merged_folder)
    splited_file_list = glob.glob(os.path.join(splited_folder,'*-slice-*.txt') )
    file_base_name_list = [os.path.basename(e) for e in splited_file_list]
    file_base_name_list = [os.path.splitext(e)[0] for e in file_base_name_list]
    file_raw_name_list = [e.split('-slice-')[0] for e in file_base_name_list]
    file_raw_name_clean = list(set(file_raw_name_list))
    merged_file_N = len(file_raw_name_clean)
    merged_file_names = []

    for k in range(merged_file_N):
        max_split_N = 0
        for name in file_base_name_list:
            name_parts = name.split('-slice-')
            if name_parts[0] == file_raw_name_clean[k]:
                n = int(name_parts[-1])
                max_split_N = max(max_split_N,n)
        max_split_N += 1
        print('max_split_N = ',max_split_N)
        merged_file_name_k = os.path.join(merged_folder,file_raw_name_clean[k]+'.txt')
        merged_file_names.append(merged_file_name_k)
        with open(merged_file_name_k,'w') as f_m_k:
            for n in range(max_split_N):
                splited_file_name_n = file_raw_name_clean[k]+'-slice-'+str(n)+'.txt'
                print(splited_file_name_n)
                splited_file_name_n = os.path.join(splited_folder,splited_file_name_n)
                with open(splited_file_name_n,'r') as f_s_n:
                    for line in f_s_n:
                        f_m_k.write(line)

    print('merge \n %s \n to %s'%(file_base_name_list,merged_file_names))
#-------------------------------------------------------------------------------
# Test Functions
#-------------------------------------------------------------------------------
def test_merge_files():
    splited_folder = '/home/x/Research/Dataset/ETH_Semantic3D_Dataset/training/tmp/bildstein_station3_xyz_intensity_rgb'
    merged_folder = '/home/x/Research/Dataset/ETH_Semantic3D_Dataset/training/tmp/bildstein_station3_xyz_intensity_rgb/merged'
    merge_spilted_files(splited_folder)


def test_split_file():
    label_file_name = '/home/x/Research/Dataset/ETH_Semantic3D_Dataset/training/tmp/bildstein_station3_xyz_intensity_rgb.labels'
    splited_folder = '/home/x/Research/Dataset/ETH_Semantic3D_Dataset/training/tmp/bildstein_station3_xyz_intensity_rgb'
    split_N = 7
    data_label_files_split(label_file_name)

def test():
    file_name = '/home/x/Research/Dataset/ETH_Semantic3D_Dataset/training/part3/sg27_station9_intensity_rgb.labels'
    file_name = '/home/x/Research/Dataset/ETH_Semantic3D_Dataset/training/part3/sg27_station9_intensity_rgb.txt'
    line_num = 222908912
    def test_islice():
        '''
        61.7775468826 sec
        '''
        num_workers = 1
        step = int(line_num / num_workers)
        for i in range(num_workers):
            with  open(file_name,'r') as f:
               # f_s = itertools.islice(f,i*step,(i+1)*step)
                f_s = itertools.islice(f,line_num - 5,None)
                for l in f_s:
                    print(i*step,'\t',l)
                    break
    def test_count_line_num():
        ''' 17.4672570229 sec 9.8191511631 '''
        N = get_total_line_num(file_name)
        print ('N = ',N)
    #test_count_line_num()
    test_islice()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get the data paths')
    parser.add_argument('--ETH_DataFolder',type=str,
                        default='/short/dh01/yx2146/Dataset/ETH_Semantic3D_Dataset/training/part0',
                        help = 'the folder contains data.txt and data.labels')
    parser.add_argument('--LabeledData_OutFolder',type=str,
                        default='/short/dh01/yx2146/Dataset/ETH_Semantic3D_Dataset/training_labeled/part0',
                        help = 'the folder to store labeled data')
    args = parser.parse_args()
    ETH_DataFolder = args.ETH_DataFolder
    LabeledData_OutFolder = args.LabeledData_OutFolder


    start_time = time.time()
    #collect_ETH3d(ETH_DataFolder,LabeledData_OutFolder)
    print('T = ',time.time() - start_time)

    print('collect_ETH3d_data main exit')
