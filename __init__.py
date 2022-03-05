bl_info = {
    "name" : "pytorch_camera",
    "author" : "close",
    "description" : "",
    "blender" : (2, 80, 0),
    "version" : (0, 0, 1),
    "location" : "",
    "warning" : "",
    "category" : "Generic"
}

import bpy
from .train_two_cam import train
from .operators import *

classes = [
    pytorch_create_camera,
    pytorch_camera_position_manual,
    pytorch_camera_position_auto,
    pytorch_camera_save_epipole,
    VIEW3D_PT_pytorch_camera]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
