from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os,sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
import x_indoor3d_util
from x_print import *

ROOM_N = 2

anno_paths = [line.rstrip() for line in open (os.path.join(BASE_DIR,'meta/anno_paths.txt')) ]
RAW_DATA_PATH = x_indoor3d_util.DATA_PATH
anno_paths = [ os.path.join(RAW_DATA_PATH,p) for p in anno_paths]

anno_paths = anno_paths[0:ROOM_N]

out_dir = os.path.join(ROOT_DIR,'x_data/standford_indoor3d')
if  not os.path.exists(out_dir):
    os.makedirs(out_dir)
    #print('mk dir: ',out_dir)

for anno_path in anno_paths:
    try:
        elements = anno_path.split('/')
        out_filename = os.path.join( out_dir, elements[-3]+'_'+elements[-2] )
        #print(elements,'\n',out_filename)
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        x_indoor3d_util.collect_point_label(anno_path,out_filename,'numpy')
    except:
        print('ERROR!')

print('end')



