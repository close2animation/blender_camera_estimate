import bpy
import torch
from .camera import camera_object
from .transform_utils import *

def train():
    print('test')
    # create camera
    cam1 = camera_object('camera_1')
    matrix = transform_Matrix_to_tensor(cam1.bpy_camera.matrix_world)
    cam1.matrix = matrix

    '''
    # create transform 
    location = torch.tensor((0, 0, 0))
    euler = torch.tensor((0, 0, 0))
    convention = 'XYZ'
    cam1.matrix = loc_rot_to_matrix(location, euler, convention)
    '''

    # apply transform to torch and bpy camera
    cam1.bpy_camera.matrix_world = transform_tensor_to_Matrix(cam1.matrix)
    cam1.bpy_mesh = bpy.data.objects['Icosphere_1']
    cam1.normalise_mesh()
    test = tensor_to_mesh_list(cam1.mesh_norm)
    create_object(test, 'test')
