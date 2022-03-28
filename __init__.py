import bpy
import sys
import subprocess
import ensurepip
import os
import imp

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

def install_libs(): 
    ensurepip.bootstrap()
    os.environ.pop("PIP_REQ_TRACKER", None)
    version = bpy.app.version_string
    version = int(version[2])
    path = sys.executable
    path = path.split('bin')[0]
    path = path + 'lib\site-packages'
    python_path = sys.executable

    subprocess.check_output([python_path, '-m', 'pip', 'install', 'opencv-python', '-t', path])
    subprocess.check_output([python_path, '-m', 'pip', 'install', 'mediapipe', '-t', path])
    subprocess.check_output([python_path, '-m', 'pip', 'install', '--ignore-installed', 'six', '-t', path])
    subprocess.check_output([python_path, '-m', 'pip', 'install', '--ignore-installed', 'attrs', '-t', path])
    subprocess.check_output([python_path, '-m', 'pip', 'install', '--ignore-installed', 'matplotlib', '-t', path])
    subprocess.check_output([python_path, '-m', 'pip', 'install', '--ignore-installed', 'torch', '-t', path])

try:
    imp.find_module('mediapipe')
    imp.find_module('mediapipe')
except ImportError:
    install_libs()

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
