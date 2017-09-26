from __future__ import  print_function
import h5py
import glob,os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--fn',help="the full path of file name to be shown")
FLAGS = parser.parse_args()
if not os.path.exists(FLAGS.fn):
    print('file not exists:\n',FLAGS.fn)


def ShowDsetAttrs(f,dset_names,m=5):
    #f = h5py.File(file_name,'r')
    for dset_name in dset_names:
        dset_name = str(dset_name)
        if dset_name in f:
            dset = f[dset_name]
            print('\ndset: ',dset_name,'  shape= ',dset.shape)
            for a in dset.attrs:
                print(a,' = \n',dset.attrs[a])
            if len(dset.shape) == 3:
                m = min(m,dset.shape[1])
                d0 = dset[0,0:m,:]
            if len(dset.shape) == 2:
                m = min(m,dset.shape[0])
                d0 = dset[0:m,:]
            print(d0)

def ShowRootAttrs(file_name):
    f = h5py.File(file_name,'r')
    for a in f.attrs:
        print(a,' = ',f.attrs[a])
    dset_names = []
    for i,dset_name in enumerate(f):
        dset_names.append(dset_name)
        if i > 10:
            break
    if i>10:
            print('dset num = %d, do not list the others'%(i+1))
    ShowDsetAttrs(f,dset_names)

def ShowData(h5f):
    dset = h5f['data']
    d = dset[0,0:3,:]
    print(d)



def ShowJson(fn):
    data =  json.load(open(fn,'r'))
    print(data)


fn = FLAGS.fn
if os.path.splitext(fn)[-1] == ".json":
    ShowJson(fn)
else:
    with h5py.File(fn,'r') as h5f:
        ShowRootAttrs(fn)
        #ShowData(h5f)

        #ShowDsetAttrs(fn,range(2005580,3005580))

