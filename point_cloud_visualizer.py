'''
display a point cloud

jiang wei
'''
import cameras
try:
    from .load_data import DepthMapDataLoader
    from .load_data import PCDDataLoader
except Exception:
    from load_data import DepthMapDataLoader
    from load_data import PCDDataLoader
import bounding_box

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from math import inf
import numpy as np

class PointCloudVisualizer():

    def __init__(self, skip=1):
        self.skip = skip

    def init_figure(self):
        fig = plt.figure()
        #add axis
        ax = fig.add_subplot(111, projection='3d')
        return fig

    def plot_pc(self, pc, fig):
        #show pc using matplotlib
        ax = plt.gca()

        pc_x = pc[:, 0]
        pc_y = pc[:, 1]
        pc_z = pc[:, 2]

        ax.scatter(pc_x[1::self.skip], pc_y[1::self.skip], pc_z[1::self.skip], c='r', marker='o', s=1)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

    def show_figure(self):
        plt.axis('equal') #preserve ratio
        #reverse x/y axis, just for visualization, the values keep the same
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.show()

    def show_pc(self, pc):
        print(pc.shape)
        fig = self.init_figure()
        self.plot_pc(pc, fig)
        self.show_figure()

    def set_skip(self, skip):
        self.skip = skip

#TO-DO: PCDVisualizer
class PCDVisualizer(PointCloudVisualizer):
    
    def __init__(self, skip=None, near=0, far=inf, rotate=0):
        if skip == None:
            PointCloudVisualizer.__init__(self)
        else:
            PointCloudVisualizer.__init__(self, skip)
        self.loader = PCDDataLoader()
        self.near = near
        self.far = far
        self.rotate = rotate
    
    def show_pc_from_pcd(self, pcd_path, bounding_box=None):
        pc = self.loader.load_pc_from_pcd(pcd_path)
        pc = self.generate_pc(pc, bounding_box=bounding_box)
        self.show_pc(pc)
        
    def generate_pc(self, in_pc, bounding_box=None):
        #point cloud unit: m
        index_pos = np.where((in_pc[: ,2] <= self.far) & (in_pc[: ,2] >= self.near))
        out_pc = in_pc[index_pos]

        if bounding_box==None:
            pass
        else:
            assert(hasattr(bounding_box, 'crop_point_cloud'))
            out_pc = bounding_box.crop_point_cloud(out_pc)

        return out_pc

    
class DepthMapVisualizaer(PointCloudVisualizer):

    def __init__(self, camera, skip=None, near=0, far=inf, rotate=0):
        if skip == None:
            PointCloudVisualizer.__init__(self)
        else:
            PointCloudVisualizer.__init__(self, skip)
        self.camera = camera
        self.loader = DepthMapDataLoader(self.camera)
        self.near = near
        self.far = far
        self.rotate = rotate

    def show_depth_map_from_depth_map_file(self, file_path):
        depth_map = self.loader.load_depth_map_from_file(file_path)
        #show depth_map
        plt.figure()
        plt.imshow(depth_map)
        plt.show()

    def show_pc_from_depth_map_file(self, file_path, bounding_box=None):
        depth_map = self.loader.load_depth_map_from_file(file_path)
        pc = self.generate_pc(depth_map, bounding_box=bounding_box)
        self.show_pc(pc)

    def generate_pc(self, depth_map, bounding_box=None):
        #point cloud unit: m
        #create an empty matrix to store the point cloud
        pc = np.zeros((self.camera.image_size[0], self.camera.image_size[1], 3))
        
        pc[:, :, 0] = np.tile(np.array([np.arange(pc.shape[0])]).T, pc.shape[1])
        pc[:, :, 1] = np.tile(np.array([np.arange(pc.shape[1])]), (pc.shape[0],1))
        pc[:, :, 2] = depth_map

        pc[:, :, 0] = (pc[:, :, 0] - self.camera.cy) * pc[:, :, 2] / self.camera.fy
        pc[:, :, 1] = (pc[:, :, 1] - self.camera.cx) * pc[:, :, 2] / self.camera.fx

        #reshape the orgnized pc to unorgnized pc
        pc = np.reshape(pc, (-1, 3))
        #now 0 position is y, 1 position is x, switch 2 cols
        pc[:, 0], pc[:, 1] = pc[:, 1], pc[:, 0].copy()

        #clip pc using near and far
        index_pos = np.where((pc[: ,2] <= self.far) & (pc[: ,2] >= self.near))
        pc = pc[index_pos]
        #crop pc using bounding box
        if bounding_box==None:
            pass
        else:
            assert(hasattr(bounding_box, 'crop_point_cloud'))
            pc = bounding_box.crop_point_cloud(pc)

        return pc

if __name__ == '__main__':
    bb1 = bounding_box.AABB(center=[0,0,2.2], half_xyz=[1, 1, 1])
    print(bb1.unit())
    
    maya_camera = cameras.MayaCamera()
    show_pc = DepthMapVisualizaer(maya_camera, skip=5, near=1, far=3)
    show_pc.show_pc_from_depth_map_file('./sample_data/test/1/images/depthRender/Cam1/mayaProject.000001.png', bounding_box=bb1)

    bb2 = bounding_box.AABB(center=[0,0,0], half_xyz=[0.2, 0.1, 1])
    vis_pcd = PCDVisualizer()
    vis_pcd.show_pc_from_pcd('./sample_data/bunny.pcd')