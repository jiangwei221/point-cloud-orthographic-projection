'''
load depth map

jiang wei
'''

import csv
import numpy as np
import cameras
from scipy import misc
from pypcd import pypcd

class PCDDataLoader():
    
    def __init__(self):
        pass

    def load_pc_from_pcd(self, pcd_path):
        #the original pypcd read the pcd file into a numpy array with customized dtype
        #here we use simple n*3 array to represent a point cloud
        pc = pypcd.PointCloud.from_path(pcd_path)
        pc2 = np.zeros((pc.pc_data.shape[0], 3))
        pc2[:,0] = pc.pc_data[:]['x']
        pc2[:,1] = pc.pc_data[:]['y']
        pc2[:,2] = pc.pc_data[:]['z']
        return pc2

class DepthMapDataLoader():
    
    #this dataloader aim to load point cloud data from depth image
    #therefore, it requires the camera params, like fx and fy

    def __init__(self, camera):
        self.camera = camera

    def remove_last_comma(self, fileobj):
        for line in fileobj:
            if line.strip()[-1] == ',':
                yield line.strip()[:-1]
            else:
                yield line.strip()

    def load_depth_map_from_csv(self, csv_path):
        reader = csv.reader(self.remove_last_comma(open(csv_path, 'r')), delimiter=',')
        x = list(reader)
        result = np.array(x).astype(float)
        assert(result.shape == self.camera.image_size)
        return result

    def load_depth_map_from_png(self, png_path):
        png_image = misc.imread(png_path, mode='L')
        print(png_image.shape, self.camera.image_size)
        assert(png_image.shape == self.camera.image_size)
        assert(hasattr(self.camera, 'convert_png_to_depth_map'))
        depth_map = self.camera.convert_png_to_depth_map(png_image)
        return depth_map

    def load_depth_map_from_file(self, file_path):
        if(file_path.endswith('.csv')):
            depth_map = self.load_depth_map_from_csv(file_path)
        elif(file_path.endswith('.png')):
            depth_map = self.load_depth_map_from_png(file_path)
        else:
            raise('unsuported format')

        return depth_map

class UBC3VDataLoader(DepthMapDataLoader):

    def __init__(self):
        DepthMapDataLoader.__init__(self, cameras.MayaCamera())

    def load_seg_image(self, seg_image_path):
        #load png image
        seg_image = misc.imread(seg_image_path, mode='RGBA')
        assert(seg_image.shape[:-1] == self.camera.image_size)
        assert(seg_image.shape[2] == 4)
        return seg_image
    
    def load_joints_gt(self, joint_gt_path):
        #load npy file
        #joint unit: m
        joints_gt = np.load(joint_gt_path)
        joints_gt /= 100 #cm to m
        return joints_gt

    def get_joints_by_index(self, joints_gt, instance_index):
        assert(instance_index < joints_gt.shape[0])
        #12:15 is the joint location
        #the whole data also contains the rotation matrix of the joint
        joints = joints_gt[instance_index, :, 12:15]
        return joints

    def load_ex_cam_list(self, ex_cam_list_path):
        #load npy file
        #ex_cam should have the shape (3, n, 2, 3)
        #3 cameras, n instances, 2 == 1 + 1, one for translation, one for rotation, 3 for x/y/z for translation or rotation
        ex_cam_list = np.load(ex_cam_list_path)
        #translation unit: m
        ex_cam_list[:, :, 0, :] /= 100 #cm to m
        assert(len(ex_cam_list.shape)==4)
        assert(ex_cam_list.shape[0]==3)
        assert(ex_cam_list.shape[2]==2)
        assert(ex_cam_list.shape[3]==3)

        return ex_cam_list

    def get_ex_cam_by_index(self, ex_cam_list, cam_index, instance_index):
        #definitions:
        #camera: maya camera
        #instance: one human model with one skeleton configuration
        #sample: captured image of one instancec by maya camera
        #every instance has 3 different maya cameras
        #and the location of the 3 cameras changes from instance to instance
        #to get the correct extrinsic parameters, one need to specify the camera id and instance id
        assert(cam_index < 3 and cam_index >= 0)
        assert(instance_index < ex_cam_list.shape[1])
        assert(len(ex_cam_list.shape)==4)
        assert(ex_cam_list.shape[0]==3)
        assert(ex_cam_list.shape[2]==2)
        assert(ex_cam_list.shape[3]==3)
        ex_cam = ex_cam_list[cam_index][instance_index]
        return ex_cam


if __name__ == '__main__':
    loader = UBC3VDataLoader()
    test_png = loader.load_depth_map_from_png('./sample_data/test/1/images/depthRender/Cam1/mayaProject.000001.png')

    loader2 = PCDDataLoader()
    test_pc = loader2.load_pc_from_pcd('./sample_data/bunny.pcd')
    test_pc = np.array(test_pc)
    print(test_pc.shape) 