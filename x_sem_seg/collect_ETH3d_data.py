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

def collect_one_file(args_list):
    data_file_name = args_list[0]
    label_file_name = args_list[1]
    out_file_name = args_list[2]
    line_num_limit = None

    print('start collecting ',data_file_name )
    with open(data_file_name,'r') as data_f, open(label_file_name,'r') as labels_f, open(out_file_name,'w')  as out_f:
        for i,e in enumerate(zip(data_f,labels_f)):
            data_line = e[0]
            data_line = remove_column_from_str(data_line.strip(),3)
            labels_line = e[1]
            out_line = data_line + '\t' + labels_line
            out_f.write(out_line)

            if line_num_limit!=None and i > line_num_limit:
                break
            if i%500000 == 0:
                print('line %d in %s'%(i,os.path.basename(data_file_name)))
    print('end writing %s'%(out_file_name) )


def collect_ETH3d(ETH_DataFolder,LabeledData_OutFolder):
    import shutil

    if not os.path.exists(ETH_DataFolder):
        print('ERROR   ETH_DataFolder not exists: ',ETH_DataFolder)
        return
    labels_file_list = glob.glob(ETH_DataFolder+'/*.labels')
    IsMultiProcess = False

    if os.path.exists(LabeledData_OutFolder):
        shutil.rmtree(LabeledData_OutFolder)
    os.makedirs(LabeledData_OutFolder)

    print(len(labels_file_list),' files detected')
    if IsMultiProcess:
        p = mp.Pool(mp.cpu_count()-0)

    for label_file_name in labels_file_list:
        label_file_basename = os.path.splitext(label_file_name)[0]
        data_file_name = label_file_basename+'.txt'
        if  not os.path.isfile(data_file_name):
            print('file not exist: ',data_file_name)
            continue
        out_file_name = os.path.basename(label_file_basename) + '_labeled.txt'
        out_file_name = os.path.join(LabeledData_OutFolder,out_file_name)
        args_list = [data_file_name,label_file_name,out_file_name]

        if IsMultiProcess:
            p.apply_async(collect_one_file,args=(args_list,))
        else:
            collect_one_file(args_list)
    if IsMultiProcess:
        p.close()
        p.join()
        print('multi process')
    else:
        print('single process')

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
    collect_ETH3d(ETH_DataFolder,LabeledData_OutFolder)
    print('T = ',time.time() - start_time)

    print('collect_ETH3d_data main exit')
