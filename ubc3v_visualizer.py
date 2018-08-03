'''
ubc3v dataset visualizer

jiang wei
'''

import cameras
from point_cloud_visualizer import DepthMapVisualizaer
from load_data import UBC3VDataLoader
import bounding_box
from transformations import transformation_mat_4x4d_from_rotation, transformation_mat_4x4d_from_translation, point_list_to_homo

import numpy as np
from numpy.linalg import inv
import math

class UBC3VVisualizer(DepthMapVisualizaer):
    
    def __init__(self):
        self.camera = cameras.MayaCamera()
        DepthMapVisualizaer.__init__(self, self.camera, skip=5)
        self.loader = UBC3VDataLoader()

    def transform_pc_to_world_space(self, pc, ex_cam):
        assert(ex_cam.shape == (2,3))
        assert(len(pc.shape) == 2)
        assert(pc.shape[1] == 3)

        default_lookat_3d = np.array([1.0 , -1.0, -1.0])

        tranlation_3d = ex_cam[0]
        rotation_3d = ex_cam[1] / 180.0 * math.pi
        rot_mat_4x4d = transformation_mat_4x4d_from_rotation(rotation_3d)
        translate_mat_4x4d = transformation_mat_4x4d_from_translation(tranlation_3d)
        transformation_mat_4x4d = np.matmul(translate_mat_4x4d, rot_mat_4x4d)

        world_pc = point_list_to_homo(pc * default_lookat_3d).dot(transformation_mat_4x4d.T)[:, :-1]
        print('world_pc', world_pc.shape)
        return world_pc

    def transform_joints_to_camera_space(self, joints, ex_cam):
        #the inverse of pc_to_world mat
        assert(ex_cam.shape == (2, 3))
        assert(joints.shape == (18, 3))
        
        default_lookat_3d = np.array([1.0 , 1.0, -1.0])
        # pc_set = pc.reshape((pc.shape[0] * pc.shape[1], 3))

        tranlation_3d = ex_cam[0]
        rotation_3d = ex_cam[1] / 180.0 * math.pi
        rot_mat_4x4d = transformation_mat_4x4d_from_rotation(rotation_3d)
        translate_mat_4x4d = transformation_mat_4x4d_from_translation(tranlation_3d)
        transformation_mat_4x4d = np.matmul(translate_mat_4x4d, rot_mat_4x4d)
        #take inverse
        inv_mat = inv(transformation_mat_4x4d)
        
        camera_joints = point_list_to_homo(joints).dot(inv_mat.T)[:, :-1]
        camera_joints = camera_joints * default_lookat_3d
        return camera_joints

    def plot_joints(self, joints, fig):
        ax = fig.gca()
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='y', marker='^', s=10)
        #draw skeleton
        bone_x = []
        bone_y = []
        bone_z = []
        edges = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 5, 9, 9, 10, 10, 11, 12, 13, 13, 14, 15, 16, 16, 17]
        for i in range(0, len(edges), 2):
            bone_x.append(joints[edges[i]][0])
            bone_y.append(joints[edges[i]][1])
            bone_z.append(joints[edges[i]][2])

            bone_x.append(joints[edges[i+1]][0])
            bone_y.append(joints[edges[i+1]][1])
            bone_z.append(joints[edges[i+1]][2])

            ax.plot(bone_x, bone_y, bone_z, c='y')

            bone_x = []
            bone_y = []
            bone_z = []

    def show_pc_w_joints(self, pc, joints):
        fig = self.init_figure()
        self.plot_pc(pc, fig)
        self.plot_joints(joints, fig)
        self.show_figure()

    def show_pc_fusion(self, pc_list, ex_cam_list):
        fig = self.init_figure()
        world_pc = []
        print(pc_list[0].shape)
        for i in range(len(pc_list)):
            world_pc.append(self.transform_pc_to_world_space(pc_list[i], ex_cam_list[i]))
        # world_pc = np.array(world_pc)
        # world_pc = np.concatenate((a, b), axis=0)
        # print(world_pc.shape)
        # world_pc = world_pc.reshape((-1, 3))
        colors = ['r', 'g', 'b']
        for i, pc in enumerate(world_pc):
            self.plot_pc(pc, fig, color=colors[i])
        # self.plot_pc(world_pc, fig)
        self.show_figure()

    def show_pc_fusion_w_joints(self, pc_list, ex_cam_list, joints):
        fig = self.init_figure()
        world_pc = []
        print(pc_list[0].shape)
        for i in range(len(pc_list)):
            world_pc.append(self.transform_pc_to_world_space(pc_list[i], ex_cam_list[i]))
        # world_pc = np.array(world_pc)
        # print(world_pc.shape)
        # print('world_pc', world_pc.shape, world_pc[0].shape)
        # world_pc = world_pc.reshape((-1, 3))
        # self.plot_pc(world_pc, fig)
        colors = ['r', 'g', 'b']
        for i, pc in enumerate(world_pc):
            self.plot_pc(pc, fig, color=colors[i])
        self.plot_joints(joints, fig)
        self.show_figure()

if __name__ == '__main__':
    bb1 = bounding_box.AABB(center=[0,0,2.2], half_xyz=[1, 1, 0.6])
    test_ubc3v = UBC3VVisualizer()
    # test_depth_map = test_ubc3v.loader.load_depth_map_from_file('../sample_data/test/1/images/depthRender/Cam1/mayaProject.000001.png')
    # test_pc = test_ubc3v.generate_pc(test_depth_map, bb1)
    # test_ubc3v.show_pc(test_pc)

    # cam_id = 1
    # instance_1d = 1

    # ex_cam_list = test_ubc3v.loader.load_ex_cam_list('../sample_data/test/1/groundtruth_cams.npy')
    # joints_gt = test_ubc3v.loader.load_joints_gt('../sample_data/test/1/groundtruth_joints.npy')

    # world_pc = test_ubc3v.transform_pc_to_world_space(test_pc, test_ubc3v.loader.get_ex_cam_by_index(ex_cam_list, 0, 0))

    # cam_joints = test_ubc3v.transform_joints_to_camera_space(test_ubc3v.loader.get_joints_by_index(joints_gt, 0), test_ubc3v.loader.get_ex_cam_by_index(ex_cam_list, 0, 0))

    # test_ubc3v.show_pc_w_joints(world_pc, test_ubc3v.loader.get_joints_by_index(joints_gt, 0))


    # test_ubc3v.show_pc_w_joints(test_pc, cam_joints)


    test_pc_fusion = []

    test_depth_map_1 = test_ubc3v.loader.load_depth_map_from_file('./sample_data/test/1/images/depthRender/Cam1/mayaProject.000001.png')
    test_depth_map_2 = test_ubc3v.loader.load_depth_map_from_file('./sample_data/test/1/images/depthRender/Cam2/mayaProject.000001.png')
    test_depth_map_3 = test_ubc3v.loader.load_depth_map_from_file('./sample_data/test/1/images/depthRender/Cam3/mayaProject.000001.png')

    test_pc_1 = test_ubc3v.generate_pc(test_depth_map_1, bb1)  
    test_pc_2 = test_ubc3v.generate_pc(test_depth_map_2, bb1)  
    test_pc_3 = test_ubc3v.generate_pc(test_depth_map_3, bb1)    

    # test_pc_fusion.append(test_pc_1)
    test_pc_fusion = np.array([test_pc_1, test_pc_2, test_pc_3])
    print(test_pc_fusion.shape)
    # test_pc_fusion = test_pc_fusion.reshape((-1, 3))
    print(test_pc_fusion[0].shape)
    print('test_pc_fusion', test_pc_fusion.shape)
    # test_pc_fusion.append(test_pc_2)
    # test_pc_fusion.append(test_pc_3)

    ex_cam_list = test_ubc3v.loader.load_ex_cam_list('./sample_data/test/1/groundtruth_cams.npy')
    joints_gt = test_ubc3v.loader.load_joints_gt('./sample_data/test/1/groundtruth_joints.npy')

    ex_cam_1 = test_ubc3v.loader.get_ex_cam_by_index(ex_cam_list, 0, 0)
    ex_cam_2 = test_ubc3v.loader.get_ex_cam_by_index(ex_cam_list, 1, 0)
    ex_cam_3 = test_ubc3v.loader.get_ex_cam_by_index(ex_cam_list, 2, 0)

    # ex_cam_fushion = np.array([  ex_cam_3])
    ex_cam_fushion = []
    ex_cam_fushion.append(ex_cam_1)
    ex_cam_fushion.append(ex_cam_2)
    ex_cam_fushion.append(ex_cam_3)

    test_ubc3v.show_pc_fusion_w_joints(test_pc_fusion, ex_cam_fushion, test_ubc3v.loader.get_joints_by_index(joints_gt, 0))