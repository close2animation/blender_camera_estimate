import bpy
import torch
from .camera import camera_object
from .transform_utils import *


def create_transfroms(location, euler, cam):
    location = torch.tensor(location)
    euler = torch.tensor(euler)
    convention = 'XYZ'
    cam.matrix = loc_rot_to_matrix(location, euler, convention)


def train():
    print('test')
    # create camera
    cam1 = camera_object('camera_2')
    cam1.matrix = Matrix_to_tensor(cam1.bpy_camera.matrix_world)
    cam1.constant = torch.tensor([-1])
    cam1.bpy_mesh = bpy.data.objects['Icosphere_1']

    cam1.update_blend_camera()
    cam1.normalise_mesh()
    cam1.project_mesh_to_camera_space()

    create_object(cam1.image_world.tolist(), 'image')
