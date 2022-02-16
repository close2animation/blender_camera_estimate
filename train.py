import bpy
import torch
from .camera import camera_object
from .transform_utils import *


def create_transfroms(location, euler, cam):
    location = torch.tensor(location)
    euler = torch.tensor(euler)
    convention = 'XYZ'
    cam.matrix = loc_rot_to_matrix(location, euler, convention)


def create_image(cam, mesh=None):
    '''
    
    '''
    if mesh:
        mesh.mesh_norm = transform_mesh_tensor(mesh, torch.inverse(cam.matrix))
    else:
        cam.normalise_mesh()
    cam.project_mesh_to_camera_space()
    return cam.image_norm
    

def train():
    print('test')
    # create camera
    cam1 = camera_object('camera_1')
    cam1.matrix = Matrix_to_tensor(cam1.bpy_camera.matrix_world)
    cam1.bpy_mesh = bpy.data.objects['Icosphere_1']
    cam1.constant = torch.tensor([-1])
    
    cam2 = camera_object('camera_2')
    cam2.matrix = Matrix_to_tensor(cam2.bpy_camera.matrix_world)
    cam1.bpy_mesh = bpy.data.objects['Icosphere_1']
    cam2.constant = torch.tensor([-1])

    # create target
    target1 = create_image(cam1) 
    target2 = create_image(cam2)



