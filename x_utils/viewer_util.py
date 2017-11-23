import numpy as np
import os


class Viewer:

    def draw_line(self,vertex1,vertex2,stride=0.001):
        dis = np.sqrt( np.sum( np.square( vertex1-vertex2)))
        line = [vertex1]
        for s in range(dis,stride):
            line.append(vertex1+s)

    def draw_2Dblock(self,zero_pos=np.array([0,0]),width=1,length=1):
        '''
        unit:m
        '''
        vertexs = [zero_pos]
        vertexs += [ vertexs[-1]+np.array([width,0])]
        vertexs += [ vertexs[-1]+np.array([0,length])]
        vertexs += [ vertexs[-1]+np.array([-width,0])]

