# -*- coding: utf-8 -*-
# transformations.py

# Copyright (c) 2006-2018, Christoph Gohlke
# Copyright (c) 2006-2018, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

'''
useful 3d vector transformation functions, some code of this file has the copyright above

jiang wei
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy import misc
import math

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

def euler_matrix(ai, aj, ak, axes='sxyz'):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> R = euler_matrix(1, 2, 3, 'syxz')
    >>> np.allclose(np.sum(R[0]), -1.34786452)
    True
    >>> R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
    >>> np.allclose(np.sum(R[0]), -0.383436184)
    True
    >>> ai, aj, ak = (4*math.pi) * (np.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)
    >>> for axes in _TUPLE2AXES.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    return M

def translation_matrix(direction):
    """Return matrix to translate by direction vector.

    >>> v = np.random.random(3) - 0.5
    >>> np.allclose(v, translation_matrix(v)[:3, 3])
    True

    """
    M = np.identity(4)
    M[:3, 3] = direction[:3]
    return M

'''
following code is wrote by jiang wei
'''

#generate the 4x4 transformation matrix from x/y/z euler angles
def transformation_mat_4x4d_from_rotation(rotate_3d):
    assert(len(rotate_3d) == 3)
    trans_mat = euler_matrix(rotate_3d[0], rotate_3d[1], rotate_3d[2])
    return trans_mat

#generate the 4x4 transformation matrix from x/y/z translation
def transformation_mat_4x4d_from_translation(translate_3d):
    assert(len(translate_3d) == 3)
    trans_mat = translation_matrix(translate_3d)
    # print(trans_mat)
    return trans_mat

#vec3 to vec4, append 1
def point_to_homo(point_3d):
    assert(len(point_3d) == 3)
    homo_pt = np.append(point_3d.copy(), [1.0])
    return homo_pt

#vec3 list to vec4 list
def point_list_to_homo(point_3d_list):
    assert(len(point_3d_list.shape) == 2)
    assert(point_3d_list.shape[1] == 3)
    homo_list = np.ones((point_3d_list.shape[0],point_3d_list.shape[1]+1))
    homo_list[:,:-1] = point_3d_list
    return homo_list

#no scale in this situation

if __name__ == '__main__':
    pass
    #given
    default_lookat_3d = np.array([0.0 ,0.0, -1.0])
    pc_3d = np.array([0.0, 0.0, 2.0])
    pc_3d_list = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]])
    print(pc_3d_list[0])
    translate_3d = np.array([1.0, 1.0, -3.0])
    rotate_3d = np.array([0.0, math.pi, 0.0])
    print(pc_3d.shape)
    #ask, what is pc in world space
    rot_mat_4x4d = transformation_mat_4x4d_from_rotation(rotate_3d)
    translate_mat_4x4d = transformation_mat_4x4d_from_translation(translate_3d)

    transformation_mat_4x4d = np.matmul(translate_mat_4x4d, rot_mat_4x4d)
    # transformation_mat_4x4d = np.matmul(rot_mat_4x4d, translate_mat_4x4d)
    print(transformation_mat_4x4d)
    print(transformation_mat_4x4d.dot(point_to_homo(pc_3d * default_lookat_3d)))

    print('times default lookat: ', pc_3d_list * default_lookat_3d)
    print( point_list_to_homo(pc_3d_list * default_lookat_3d).dot(transformation_mat_4x4d.T)[:, :-1] )