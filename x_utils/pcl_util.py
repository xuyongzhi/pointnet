import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR,'data')

class GLOBAL_PARA:
    all_catigories = ['ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter']
    good1_path = os.path.join(DATA_DIR,'stanford_indoor3d_globalnormedh5_stride_0.5_step_1_4096/obj_file/Area_6_office_25_stride_0.5_step_1_random_4096_globalnorm/all_single')



class PCL_VIEWER:

    def __init__(self):
        print('')


    @staticmethod
    def get_obj_point_num(obj_fn):
        with open(obj_fn,'r') as obj_f:
            i = 0
            for i,_ in  enumerate(obj_f):
                pass
            return i+1
    @staticmethod
    def get_obj_fileds_num(obj_fn):
        with open(obj_fn,'r') as obj_f:
            for line in  obj_f:
                line = line.strip()
                elements = line.split()
                if elements[0] == 'v':
                    return len(elements) - 1
                else:
                    return -1


    @staticmethod
    def get_pcl_head_str(obj_fn):
        fields_num = PCL_VIEWER.get_obj_fileds_num(obj_fn)
        point_num = PCL_VIEWER.get_obj_point_num(obj_fn)
        head_str = 'VERSION .7\n'
        if fields_num == 3:
            head_str += 'FIELDS x y z'
            head_str += 'SIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n'
        elif fields_num == 6:
            head_str += 'FIELDS x y z rgb\n'
            head_str += 'SIZE 4 4 4 1\nTYPE F F F U\nCOUNT 1 1 1 3\n'
        head_str += 'WIDTH %d\nHEIGHT 1\nPOINTS %d\nDATA ascii\n'%(point_num,point_num)
        return head_str

    @staticmethod
    def obj_to_pcl(obj_fn):
        pcl_fn = os.path.splitext(obj_fn)[0] + '.pcd'
        head_str = PCL_VIEWER.get_pcl_head_str(obj_fn)
        with open(obj_fn,'r') as obj_f, open(pcl_fn,'w') as pcl_f:
            pcl_f.write(head_str)
            for line in obj_f:
                line = line.strip()
                elements = line.split()
                if elements[0] == 'v':
                    elements = elements[1:len(elements)]
                    new_line = '  '.join(elements) + '\n'
                    pcl_f.write(new_line)
            print('gen file: '+os.path.basename(pcl_fn) )


if __name__ =='__main__':
    for cat in GLOBAL_PARA.all_catigories:
        obj_fn = os.path.join(GLOBAL_PARA.good1_path,'gt_'+cat+'.obj')
        PCL_VIEWER.obj_to_pcl(obj_fn)
