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

SAMPLING_BIN = os.path.join(BASE_DIR, 'third_party/mesh_sampling/build/pcsample')

SAMPLING_POINT_NUM = 2048
SAMPLING_LEAF_SIZE = 0.005

MODELNET40_PATH = '../datasets/modelnet40'
def export_ply(pc, filename):
	vertex = np.zeros(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
	for i in range(pc.shape[0]):
		vertex[i] = (pc[i][0], pc[i][1], pc[i][2])
	ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices'])])
	ply_out.write(filename)

# Sample points on the obj shape
def get_sampling_command(obj_filename, ply_filename):
    cmd = SAMPLING_BIN + ' ' + obj_filename
    cmd += ' ' + ply_filename
    cmd += ' -n_samples %d ' % SAMPLING_POINT_NUM
    cmd += ' -leaf_size %f ' % SAMPLING_LEAF_SIZE
    return cmd

# --------------------------------------------------------------
# Following are the helper functions to load MODELNET40 shapes
# --------------------------------------------------------------

# Read in the list of categories in MODELNET40
def get_category_names():
    shape_names_file = os.path.join(MODELNET40_PATH, 'shape_names.txt')
    shape_names = [line.rstrip() for line in open(shape_names_file)]
    return shape_names

# Return all the filepaths for the shapes in MODELNET40
def get_obj_filenames():
    obj_filelist_file = os.path.join(MODELNET40_PATH, 'filelist.txt')
    obj_filenames = [os.path.join(MODELNET40_PATH, line.rstrip()) for line in open(obj_filelist_file)]
    print('Got %d obj files in modelnet40.' % len(obj_filenames))
    return obj_filenames

# Helper function to create the father folder and all subdir folders if not exist
def batch_mkdir(output_folder, subdir_list):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for subdir in subdir_list:
        if not os.path.exists(os.path.join(output_folder, subdir)):
            os.mkdir(os.path.join(output_folder, subdir))

# ----------------------------------------------------------------
# Following are the helper functions to load save/load HDF5 files
# ----------------------------------------------------------------

# Write numpy array data and label to h5_filename
def save_h5_data_label_normal(h5_filename, data, label, normal,
		data_dtype='float32', label_dtype='uint8', noral_dtype='float32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'normal', data=normal,
            compression='gzip', compression_opts=4,
            dtype=normal_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()


# Write numpy array data and label to h5_filename
def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()

# Read numpy array data and label from h5_filename
def load_h5_data_label_normal(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    normal = f['normal'][:]
    return (data, label, normal)

# Read numpy array data and label from h5_filename
def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)

# Read numpy array data and label from h5_filename
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

# ----------------------------------------------------------------
# Following are the helper functions to load save/load PLY files
# ----------------------------------------------------------------

# Load PLY file
def load_ply_data(filename, point_num):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data[:point_num]
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

# Load PLY file
def load_ply_normal(filename, point_num):
    plydata = PlyData.read(filename)
    pc = plydata['normal'].data[:point_num]
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

# Make up rows for Nxk array
# Input Pad is 'edge' or 'constant'
def pad_arr_rows(arr, row, pad='edge'):
    assert(len(arr.shape) == 2)
    assert(arr.shape[0] <= row)
    assert(pad == 'edge' or pad == 'constant')
    if arr.shape[0] == row:
        return arr
    if pad == 'edge':
        return np.lib.pad(arr, ((0, row-arr.shape[0]), (0, 0)), 'edge')
    if pad == 'constant':
        return np.lib.pad(arr, ((0, row-arr.shape[0]), (0, 0)), 'constant', (0, 0))



# xyz
# split data for visualization
def split_along_dim(filename,dim,start,end):
    #MAXN = 20
    if not os.path.exists(filename):
        print('file does not exist: ',filename)
        exit()
    path = os.path.dirname(filename)
    elements = os.path.basename(filename).split('.')
    base_name = elements[0]
    file_format = elements[-1]

    out_filename = base_name+'_split_'+str(dim)+'_'+str(start)+'_'+str(end)+'.txt'
    out_filename = os.path.join(path,out_filename)
    #print(path,'\n',base_name,'\n',out_filename)

    if file_format == 'txt':
        valid_cols = range(0,6)
    elif file_format == 'obj':
        valid_cols = range(1,7)
    else:
        print('file format error:',file_format)

    data = np.loadtxt(filename,usecols=valid_cols)
    #print(data)
    #data = data[0:MAXN,:]
    dim_min = np.amin(data[:,dim],axis=0)
    dim_max = np.amax(data[:,dim],axis=0)
    dim_scope = dim_max - dim_min
    dim_start = dim_min + dim_scope*start
    dim_end = dim_min + dim_scope*end
    print('dim_min=',dim_min,'\ndim_max=',dim_max)
    IsKeep = np.nonzero(  [k>=dim_start and k<=dim_end for k in data[:,dim]])
    data_splited = data[IsKeep[0],:]

    print('data.shape= ',data.shape,'\nsplited shape= ',data_splited.shape)

    np.savetxt(out_filename,data_splited,delimiter='  ',fmt='%8.3f')
    print('save :',out_filename)
    if file_format == 'txt':
        txt2obj(filename)
    txt2obj(out_filename)


def read_obj_data(filename):
    cols = range(1,7)
    data = np.loadtxt(filename,delimiter=' ',usecols=cols)
    #print(data)
    #print(data.shape)
    return data

def txt2obj(filename):
    name_format = os.path.basename(filename).split('.')
    name = name_format[0]
    format_txt = name_format[-1]
    if not format_txt == 'txt':
        print('input format is not txt: ',format_txt)
        exit()
    out_filename = name+'.obj'
    path = os.path.dirname(filename)
    out_filename = os.path.join(path,out_filename)

    out_f = open(out_filename,'w')
    with open(filename,'r') as f:
        for line in f:
            #line = line.rstrip()
            line = 'v   ' + line
            out_f.write(line)
    out_f.close()
    print('save: ',out_filename)


def WriteFileList(folder_path,format,out_file_name,prefix):
    file_list = glob.glob(folder_path+'/*.'+format)
    print(str(len(file_list))+' items found')
    file = open(out_file_name,'w')
    for i,item in enumerate(file_list):
        item = os.path.basename(item)
        item = os.path.join(prefix,item)
        if i>0:
            item = '\n'+item
        file.write(item)


'''
    ETH Semantic3D data preparation
'''
def remove_intensity(in_filename,out_filename):
    '''
    in_filename: x y z intensity r g b
    out_filename: x y z r g b
    '''
    Data_In = np.loadtxt(in_filename)
    if not Data_In.shape[1] == 7:
        print('data shape wrong: ',Data_In.shape)
    Data_out = Data_In[:,0:3]

def count_array(array,target):
    count = array == target
    num = np.count_nonzero(count)
    return num

def get_ETH_labels():
    ETH_labels = {0: 'unlabeled points', 1: 'man-made terrain', 2: 'natural terrain', 3: 'high vegetation', 4: 'low vegetation',
                  5: 'buildings', 6: 'hard scape', 7: 'scanning artefacts', 8: 'cars'}
    return ETH_labels

def count_line(in_queue,out_count,line_num_limit,OneFileProcessNum):
    dis_step = get_dis_step(line_num_limit)
    while True:
        item = in_queue.get()
        num_line,line = item

        #exit signal
        if line==None:
            #print('return:               line==None exit ')
            return
        line = int(line.strip())
        compare = np.array( [line == e for e in range(0,9)] )
        for i in range(len(out_count)):
            out_count[i] += compare[i]
        #if num_line % dis_step == 0 or num_line >= line_num_limit:
            #print('count line ',num_line, 'queue_size = ',in_queue.qsize())

        # in the last OneFileProcessNum items, every process handle one item
        if line_num_limit!=None and num_line >= line_num_limit-OneFileProcessNum+1:
            print('return:           num_line=%d >= line_num_limit-OneFileProcessNum+1=%d '%(num_line,line_num_limit-OneFileProcessNum+1))
            return


def count_onefile_multiprocess(count_onefile_args):
    '''
    There is still some problems:
        (1) way too slow
        (2) can miss some lines
    '''
    from multiprocessing import Process, Manager,Array
    file_name = count_onefile_args[0]
    line_num_limit = count_onefile_args[1]
    OneFileProcessNum = count_onefile_args[2]

    dis_step = get_dis_step(line_num_limit)

    num_workers = OneFileProcessNum
    manager = Manager()
    count_res = Array('i',[0]*9)
    line_queue = manager.Queue(num_workers)

    pool = []
    for i in xrange(num_workers):
        p = Process(target=count_line,args=(line_queue,count_res,line_num_limit,OneFileProcessNum))
        p.start()
        pool.append(p)

    total_N = 0
    with open(file_name) as f:
        #print('A file opened')
        iters = itertools.chain(f,(None,)*num_workers)
        for num_and_line in enumerate(iters):
            line_queue.put(num_and_line)
            #if num_and_line[0] % dis_step == 0 or num_and_line[0]>=line_num_limit:
                #print ('put line ',num_and_line[0], 'in process queue')
            if line_num_limit!=None and num_and_line[0] >= line_num_limit:
                #print('break at line ',num_and_line[0])
                break
        total_N = num_and_line[0] - num_workers
    print('              finished puting all lines in process queue ')
    for p in pool:
        p.join()

    count = count_res[:]
    info_str = get_conclu_str(file_name,total_N,count)

    return  info_str


def get_dis_step(line_num_limit):
    if line_num_limit == None:
        dis_step = 10000*500
    else:
        dis_step = int( line_num_limit / 3)
        #dis_step = 1
    return dis_step

def get_conclu_str(file_name,total_N,count):
    head_str =  '\n' + os.path.basename(file_name) + ' N = ' + str(total_N) + '\n'
    num_str = '     '.join(str(i)+': '+str(int(e)) for i, e in enumerate(count)) + '\n'
    rate_str = '   '.join('%d: %0.3f'%(i,e/total_N) for i,e in enumerate(count) )
    info_str = head_str+num_str+rate_str + '\n'
    return info_str

def count_onefile(count_onefile_args):
    file_name = count_onefile_args[0]
    line_num_limit = count_onefile_args[1]

    file = open(file_name,'r')
    count = np.zeros(9)
    i = 0
    dis_step = get_dis_step(line_num_limit)

    for i,line in enumerate(file):
        line = int(line.strip())
        compare = np.array( [line == e for e in range(0,9)] )
        count += compare

        #if i % dis_step == 0 or i >= line_num_limit:
            #print('counting line ',i,'  ',count)
        if line_num_limit!=None and i>=line_num_limit:
            #print('break at line ',i)
            break
    total_N = i+1
    info_str = get_conclu_str(file_name,total_N,count)

    return info_str


def count_labels_num(labels_folder,out_file_name):
    '''
    For reading multi files, multi process is useful
    For reading within one file, one process is much faster than multi file! Maybe my code is not good.
    '''
    label_file_list = os.path.join(labels_folder,'*.labels')
    file_list = glob.glob(label_file_list)
    out_file = open(out_file_name,'w')
    out_file.write( str(get_ETH_labels())+'\n' )

    FilesProcessNum = 6
    OneFileProcessNum = 0   # <=0 no multi
    line_num_limit = 1000

    print('\nFilesProcessNum = ',FilesProcessNum)
    print('OneFileProcessNum = ',OneFileProcessNum)
    print('line_num_limit = ',line_num_limit)

    count_onefile_args_list = itertools.product(file_list,[line_num_limit],[OneFileProcessNum])

    if FilesProcessNum>0:
        p = mp.Pool()

        if OneFileProcessNum >0:
            result = p.imap_unordered(count_onefile_multiprocess,count_onefile_args_list)
        else:
            result = p.imap_unordered(count_onefile,count_onefile_args_list)
    else:
        result = []
        for file_name in file_list:
            #print(file_name)
            count_onefile_args = (file_name,line_num_limit,OneFileProcessNum)
            if OneFileProcessNum >0:
                r = count_onefile_multiprocess(count_onefile_args)
            else:
                r = count_onefile(count_onefile_args)
            result.append(r)

    for r in result:
        print(r)
        out_file.write(r)
    out_file.close()


def clean_label_files_list(label_files_list):
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

def gen_rawETH_to_h5(label_files_glob,line_num_limit=None):
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
    data_files_list, h5_files_list = clean_label_files_list(label_files_list)
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
                    add_buf(xyz_dset,data_buf[:,0:3],start,end)
                    add_buf(intensity_dset,data_buf[:,3:4],start,end)
                    add_buf(color_dset,data_buf[:,4:7],start,end)
                    add_buf(label_dset,label_buf[:,0:1],start,end)
                    h5_f.flush()

                if line_num_limit != None and k+1 >= line_num_limit:
                    print('break at k= ',k)
                    break

            add_buf_all(h5_f,xyz_dset,intensity_dset,color_dset,label_dset,data_buf,label_buf,k,buf_rows)
            cut_redundance(xyz_dset,k+1)
            cut_redundance(intensity_dset,k+1)
            cut_redundance(color_dset,k+1)
            cut_redundance(label_dset,k+1)

            print('having read %d lines from %s \n'%(k+1,data_fn))
            #print('h5 file line num = %d'%(xyz_dset.shape[0]))

def cut_redundance(dset,file_rows):
    if dset.shape[0] > file_rows:
        dset.resize((file_rows,dset.shape[1]))

def add_buf_all(h5_f,xyz_dset,intensity_dset,color_dset,label_dset,data_buf,label_buf,k,buf_rows):
    k_buf = k%buf_rows
    start = int(k/buf_rows)*buf_rows
    end = k+1
    #print( 'start = %d, end = %d'%(start,end))
    add_buf(xyz_dset,data_buf[0:k_buf+1,0:3],start,end)
    add_buf(intensity_dset,data_buf[0:k_buf+1,3:4],start,end)
    add_buf(color_dset,data_buf[0:k_buf+1,4:7],start,end)
    add_buf(label_dset,label_buf[0:k_buf+1,0:1],start,end)
    h5_f.flush()
    #print('flushing k = ',k)

def add_buf(dset,new_data,start,end):
    if dset.shape[0] < end:
        dset.resize((end,dset.shape[1]))
    dset[start:end,:] = new_data

#-------------------------------------------------------------------------------
# TEST functions
#-------------------------------------------------------------------------------

def test_split_along_dim():
    filename = '/home/x/Research/pointnet/data/Stanford3dDataset_v1.2_Aligned_Version/Area_6/conferenceRoom_1/test.txt'
    filename = '/home/x/Research/pointnet/sem_seg/log6/dump/Area_6_office_10_pred.obj'
    #split_along_dim(filename,2,0,0.7)
    #read_obj_data(filename)

def test_WriteFileList():
    folder_path = '/short/dh01/yx2146/Dataset/ETH_Semantic3D_Dataset/reduced_test'
    out_file_name = '/short/dh01/yx2146/pointnet/x_sem_seg/meta/ETH_reduced_test.txt'
    prefix='../Dataset/ETH_Semantic3D_Dataset/reduced_test'
    #WriteFileList(folder_path,'txt',out_file_name,prefix)

def test_count_labels_num():
    start_time = time.time()
    labels_folder = '/home/x/Research/Dataset/ETH_Semantic3D_Dataset/training'
    out_file_name = os.path.join( os.path.dirname(labels_folder),'count_training_labels_A.txt' )
    count_labels_num(labels_folder,out_file_name)
    T = time.time() - start_time
    print('\nT = ',T)

def test_gen_rawETH_to_h5(ETH_raw_labels_glob=None):
    if ETH_raw_labels_glob == None:
        labels_folder = '/home/x/Research/Dataset/ETH_Semantic3D_Dataset/training/part_A'
        labels_folder = '/short/dh01/yx2146/Dataset/ETH_Semantic3D_Dataset/training/part_A'
        #labels_folder = '/other/ETH_Semantic3D_Dataset/training/part_A'
        ETH_raw_labels_glob = os.path.join(labels_folder,'*.labels')
    line_num_limit = 1300005
    gen_rawETH_to_h5(ETH_raw_labels_glob)




if __name__ == '__main__':
    ''' test functions
    '''
    parser = argparse.ArgumentParser(description='input the ETH path')
    parser.add_argument('--ETH_raw_labels_glob',type = str,
                        default='/short/dh01/yx2146/Dataset/ETH_Semantic3D_Dataset/training/part_A/AA/*.labels')
    FLAGS = parser.parse_args()

    start = time.time()
    test_gen_rawETH_to_h5(FLAGS.ETH_raw_labels_glob)
    print('T = ',time.time()-start)
    print('\n OK')


