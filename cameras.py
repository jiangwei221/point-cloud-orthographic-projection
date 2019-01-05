'''
depth camera class

jiang wei
'''

import numpy as np

class BaseCamera():

    '''
    note: image size is (num_rows, num_cols)
    '''
    
    def __init__(self, image_size, cx, cy, fx, fy, k1=None, k2=None, k3=None, p1=None, p2=None):
        assert(len(image_size) == 2)
        self.image_size = image_size #film size
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2

    def name(self):
        return 'BaseCamera'

class KinectCamera(BaseCamera):

    '''
    Kinect V2
    '''
    
    def __init__(self):
        BaseCamera.__init__(self, image_size=(424, 512), cx=254.878, cy=205.395, fx=365.456, fy=365.456, k1=0.0905474, k2=-0.26819, k3=0.0950862, p1=0.0, p2=0.0)

    def name(self):
        return 'KinectCamera V2'

class MayaCamera(BaseCamera):

    '''
    simulated kinect depth camera in Maya renderer(used in ubc3v)
    note:
    % The 8-bit band encompasses the range 50 to 800 cm.
    output_args = (double(input_args(:, :, 1))./255 .* (800-50) + 50)*1.03;
    '''

    def __init__(self):
        BaseCamera.__init__(self, image_size=(424, 512),cx=256, cy=212, fx=368.096588, fy=368.096588)

    def convert_png_to_depth_map(self, png_image):
        assert(png_image.shape == self.image_size)
        #convert 0 to 255
        index_pos = np.where(png_image==0)
        png_image[index_pos] = 255
        depth_map = (png_image/255.0 * (800 - 50) + 50) / 100
        return depth_map

    def name(self):
        return 'MayaCamera'

class HandsegCamera(BaseCamera):

    '''
    simulated kinect depth camera in Maya renderer(used in ubc3v)
    note:
    % The 8-bit band encompasses the range 50 to 800 cm.
    output_args = (double(input_args(:, :, 1))./255 .* (800-50) + 50)*1.03;
    '''

    def __init__(self):
        BaseCamera.__init__(self, image_size=(256, 256),cx=128, cy=128, fx=368.096588, fy=368.096588)

    def convert_png_to_depth_map(self, png_image):
        assert(png_image.shape == self.image_size)
        #convert 0 to 255
        index_pos = np.where(png_image==0)
        png_image[index_pos] = 255
        depth_map = (png_image/255.0 * (800 - 50) + 50) / 100
        return depth_map

    def name(self):
        return 'HandsegCamera'