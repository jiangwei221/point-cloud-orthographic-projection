'''
bounding box class

jiang wei
'''

import numpy as np

class BaseBoundingBox():

    def __init__(self, center):
        #center, absolute location in world coord
        assert(len(center)==3)
        self.center = center

    def name(self):
        return 'BaseBoundingBox'

    def unit(self):
        #bounding box unit: m
        return 'm'

class AABB(BaseBoundingBox):

    def __init__(self, center, half_xyz):
        BaseBoundingBox.__init__(self, center)
        assert(len(half_xyz)==3)
        self.half_xyz = half_xyz

        self.hi_x = self.center[0] + self.half_xyz[0]
        self.low_x = self.center[0] - self.half_xyz[0]
        self.hi_y = self.center[1] + self.half_xyz[1]
        self.low_y = self.center[1] - self.half_xyz[1]
        self.hi_z = self.center[2] + self.half_xyz[2]
        self.low_z = self.center[2] - self.half_xyz[2]

        self.width = self.half_xyz[0] * 2
        self.height = self.half_xyz[1] * 2
        self.depth = self.half_xyz[2] * 2

    def name(self):
        return 'AABB'

    def crop_point_cloud(self, pc, keep_inside=True):
        #assume the input data is unorgnized
        assert(len(pc.shape)==2)
        assert(pc.shape[1]==3)

        index_pos = np.where((pc[:, 0] <= self.hi_x) & (pc[:, 0] >= self.low_x) & 
                            (pc[:, 1] <= self.hi_y) & (pc[:, 1] >= self.low_y) & 
                            (pc[:, 2] <= self.hi_z) & (pc[:, 2] >= self.low_z))

        return pc[index_pos]