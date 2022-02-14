import bpy
import torch
from .transform_utils import *

class camera_object():
    def __init__(self, camera_name):
        self.matrix = torch.tensor([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        self.camera_name = camera_name 
        self.bpy_object = None
        self.import_camera()

    def import_camera(self):
        '''
        dont wanna import an object over an over
        so just assigning for now 
        '''
        self.bpy_object = bpy.data.objects[self.camera_name]


