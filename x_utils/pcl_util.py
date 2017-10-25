import os


class PCL_VIEWER:

    def __init__(self):
        print('')


    @staticmethod
    def get_obj_point_num(obj_fn):
        with open(obj_fn,'r') as obj_f:
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


if __name__ =='__main__':
    obj_fn = '/home/y/Research/pointnet/data/stanford_indoor3d_globalnormedh5_stride_0.5_step_1_4096/obj_file/Area_3_storage_2_stride_0.5_step_1_random_4096_globalnorm/Z/raw_colored.obj'
    PCL_VIEWER.obj_to_pcl(obj_fn)
