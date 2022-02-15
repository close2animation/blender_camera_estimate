import bpy
import torch
from .transform_utils import *

class camera_object():
    def __init__(self, camera_name):
        self.matrix = torch.tensor([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        self.bpy_camera = None
        self.bpy_mesh = None
        self.mesh_norm = None
        self.import_camera(camera_name)      

    def import_camera(self, camera_name):
        '''
        dont wanna import an object over an over
        so just assigning for now 
        '''
        self.bpy_camera = bpy.data.objects[camera_name]

    def normalise_mesh(self):
        '''
        transforms the assgined mesh to the camera's normalised space. 
        normalised meaning without rotation or translation.
        '''
        mesh = mesh_space_to_world(self.bpy_mesh.data,
                                   self.bpy_mesh.matrix_world)
        mesh = mesh_list_to_tensor(mesh)
        self.mesh_norm = transform_mesh_tensor(mesh, torch.inverse(self.matrix))


