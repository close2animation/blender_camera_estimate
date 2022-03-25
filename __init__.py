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
from .operators import *

classes = [
    pytorch_create_camera,
    pytorch_camera_position_pass1,
    pytorch_camera_position_pass2,
    pytorch_create_animation,
    pytorch_camera_save_epipole,
    VIEW3D_PT_load_data,
    VIEW3D_PT_pytorch_camera,
    OT_load_data,
    My_settings]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.my_tool = bpy.props.PointerProperty(type=My_settings)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.my_tool
