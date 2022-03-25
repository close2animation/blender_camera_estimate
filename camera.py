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
        self.constant = None
        self.image_norm = None
        self.image_world = None
        self.bpy_camera = bpy.data.objects[camera_name]   


    def normalise_mesh(self):
        '''
        transforms the assgined mesh to the camera's normalised space. 
        normalised meaning without rotation or translation.
        '''
        mesh = mesh_space_to_world(self.bpy_mesh.data,
                                   self.bpy_mesh.matrix_world)                           
        mesh = vert_list_to_tensor(mesh)
        self.mesh_norm = transform_mesh_tensor(mesh, torch.inverse(self.matrix))


    def update_blend_camera(self):
        '''
        applys camera paramaters to blender camera so we can see visually
        '''
        self.bpy_camera.matrix_world = tensor_to_Matrix(self.matrix)

        c = self.constant.detach().numpy()
        mesh = self.bpy_camera.data 
        for vert in mesh.vertices:
            vert.co.z = c
        # this will error if vertex order not kept
        mesh.vertices[7].co.z = 0 


    def project_mesh_to_camera_space(self):
        '''
        projects normalised mesh points onto camera plane. 
        '''
        c = self.constant
        K = torch.tensor([[c, 0, 0, 0],
                          [0, c, 0, 0],
                          [0, 0, 1, 0]]).float()
       
        points = torch.tensor([]).float()
        for vert in self.mesh_norm:
            x = torch.matmul(K, vert)
            x = torch.divide(x, x[2])
            x[2] = -x[2]            
            points = torch.cat((points, x))

        self.image_norm = points.reshape((self.mesh_norm.shape[0],3))
        #self.image_world = transform_mesh_tensor(self.image_norm, self.matrix)


