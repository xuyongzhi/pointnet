from __future__ import print_function
import os,sys,glob
import numpy as np



BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH =os.path.join(ROOT_DIR, 'data/Stanford3dDataset_v1.2_Aligned_Version')

g_classnames = [l.rstrip() for l in open( os.path.join(BASE_DIR,'meta/class_names.txt') )]
g_class2label = {cls:i for i,cls in enumerate(g_classnames)}
g_class2color = {'ceiling':	[0,255,0],
                 'floor':	[0,0,255],
                 'wall':	[0,255,255],
                 'beam':        [255,255,0],
                 'column':      [255,0,255],
                 'window':      [100,100,255],
                 'door':        [200,200,100],
                 'table':       [170,120,200],
                 'chair':       [255,0,0],
                 'sofa':        [200,100,100],
                 'bookcase':    [10,200,100],
                 'board':       [200,200,200],
                 'clutter':     [50,50,50]}
g_label2color = {g_class2label[cls]:g_class2color[cls] for cls in g_classnames}


def collect_point_label(anno_path,out_filepath,file_format):
    '''
    collect all instance files of one room to a single file
    XYZRGB -> XYZRGBL
    args:
        anno_path:input path
        out_filepath: out path
        file_format: 'txt' or 'numpy'
    '''
    points_list = []

    for instance_file in glob.glob( os.path.join(anno_path,'*.txt') ):
        #print(instance_file)
        cls = os.path.basename(instance_file).split('_')[0]
        #print('\ncls = ',cls)
        points = np.loadtxt(instance_file)
        labels = np.ones((points.shape[0],1))*g_class2label[cls]
        points_label = np.concatenate( (points,labels),1  )
        points_list.append(points_label)
        #print('p:',points.shape)
        #print('p_l: ',points_label.shape)

    points_all = np.concatenate(points_list,0)

    if file_format == 'txt':
        f_txt = open(out_filepath+'.txt','w')
        step = points_all.shape[1]
        for i in range(points_all.shape[0]):
            j = i*step
            str = '%f %f %f %d %d %d %d\n'%(\
                        points_all[i,0],\
                        points_all[i,1],\
                        points_all[i,2],\
                        points_all[i,3],\
                        points_all[i,4],\
                        points_all[i,5],\
                        points_all[i,6]\
                        )
            f_txt.write(str)
        f_txt.close()
    elif file_format == 'numpy':
        f_numpy = np.save(out_filepath+'.numpy',points_all)
    else:
        print('wrong file format: ',file_format)
    print('list len=',len(points_list))
    print(points_all.shape)


collect_point_label('/home/x/Research/pointnet/data/Stanford3dDataset_v1.2_Aligned_Version/Area_1/conferenceRoom_1/Annotations',\
                    '/home/x/Research/pointnet/x_data/standford_indoor3d/Area_1_conferenceRoom_1',\
                    'numpy')

print('end')
