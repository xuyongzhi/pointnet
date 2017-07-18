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

def count_line(in_queue,out_count):
    while True:
        if in_queue.empty():
            return
        item = in_queue.get()
        num_line,line = item

        #exit signal
        if line==None:
            return
        line = int(line.strip())
        compare = np.array( [line == e for e in range(0,9)] )
        for i in range(len(out_count)):
            out_count[i] += compare[i]



def count_onefile_multiprocess(file_name,line_num_limit,OneFileProcessNum):
    from multiprocessing import Process, Manager,Array
    import itertools

    num_workers = OneFileProcessNum
    manager = Manager()
    count_res = Array('i',[0]*9)
    line_queue = manager.Queue(num_workers)

    pool = []
    for i in xrange(num_workers):
        p = Process(target=count_line,args=(line_queue,count_res))
        p.start()
        pool.append(p)

    total_N = 0

    with open(file_name) as f:
        #print('A file opened')
        iters = itertools.chain(f,(None,)*num_workers)
        for num_and_line in enumerate(iters):
            line_queue.put(num_and_line)
            #print ('put line ',num_and_line[0])
            if num_and_line[0] > line_num_limit:
                break
        total_N = num_and_line[0]
    for p in pool:
        p.join()

    count = count_res[:]
    info_str = get_conclu_str(file_name,total_N,count)

    return  info_str


def get_conclu_str(file_name,total_N,count):
    head_str =  '\n' + os.path.basename(file_name) + ' N = ' + str(total_N) + '\n'
    num_str = '    '.join(str(int(e)) for e in count) + '\n'
    rate_str = '  '.join('%0.1f'%(e/total_N) for e in count ) + '\n'
    info_str = head_str+num_str+rate_str
    return info_str

def count_onefile(file_name,line_num_limit):
    file = open(file_name,'r')
    count = np.zeros(9)
    i = 0

    for i,line in enumerate(file):
        line = int(line.strip())
        compare = np.array( [line == e for e in range(0,9)] )
        count += compare
        if i>line_num_limit:
            break
    total_N = i+1
    info_str = get_conclu_str(file_name,total_N,count)

    print(info_str)
    return info_str


def count_labels_num(labels_folder,out_file_name):
    label_file_list = os.path.join(labels_folder,'bildstein_station1_xyz_intensity_rgb.labels')
    file_list = glob.glob(label_file_list)
    out_file = open(out_file_name,'w')
    out_file.write( str(get_ETH_labels())+'\n' )

    FilesProcessNum = 0
    OneFileProcessNum = 5   # <=0 no multi

    if FilesProcessNum>0:
        p = mp.Pool()
        result = p.imap_unordered(count_onefile_multiprocess,file_list)
        #result = p.imap_unordered(count_onefile,file_list)
    else:
        result = []
        line_num_limit = 100
        for file_name in file_list:
            print(file_name)
            if OneFileProcessNum >0:
                r = count_onefile_multiprocess(file_name,line_num_limit,OneFileProcessNum)
            else:
                r = count_onefile(file_name,line_num_limit)
            result.append(r)

    for r in result:
        print(r)
        out_file.write(r)
    out_file.close()


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
    out_file_name = os.path.join( os.path.dirname(labels_folder),'training_labels_count.txt' )
    count_labels_num(labels_folder,out_file_name)
    T = time.time() - start_time
    print('T = ',T)


if __name__ == '__main__':
    ''' test functions
    '''
    test_count_labels_num()


    print('\n OK')


