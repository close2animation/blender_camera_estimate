import bpy
import torch
from .camera import camera_object
from .transform_utils import *

def train():
    print('test')
    # create camera
    cam1 = camera_object('camera_1')
    cam1.bpy_object

    # create transform 
    location = torch.tensor((0, 0, 0))
    euler = torch.tensor((0, 0, 0))
    convention = 'XYZ'

    # apply transform to torch and bpy camera
    cam1.matrix = loc_rot_to_matrix(location, euler, convention)
    cam1.bpy_object.matrix_world = tensor_to_Matrix(cam1.matrix)
    print(tensor_to_Matrix(cam1.matrix))
    print(Matrix_to_tensor(cam1.bpy_object.matrix_world))