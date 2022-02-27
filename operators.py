import bpy
import torch
from.train_two_cam import *
import bpy
from mathutils import Matrix


class pytorch_create_camera(bpy.types.Operator):
    bl_idname = "pytorch.create_camera"
    bl_label = "pytorch camera"
    bl_options = {'REGISTER','UNDO'}


    def execute(self, context):
        dir = bpy.utils.user_resource('SCRIPTS', "addons")
        path = f'{dir}//pytorch_camera//camera.obj'
        for i in range(2):
            bpy.ops.object.select_all(action='DESELECT')
            bpy.ops.import_scene.obj(filepath=path, split_mode='OFF')
            bpy.context.selected_objects[0].name = f'camera_{i}'

        return {'FINISHED'}


class pytorch_camera_position_manual(bpy.types.Operator):
    bl_idname = "pytorch.camera_position_manual"
    bl_label = "pytorch camera"
    bl_options = {'REGISTER','UNDO'}

    def execute(self, context):
        # create image data
        images1 = np.load('C://Users//Juan//OneDrive//transfer files//images_norm1.npy')
        images2 = np.load('C://Users//Juan//OneDrive//transfer files//images_norm2.npy')

        images1 = torch.from_numpy(images1).float()
        images2 = torch.from_numpy(images2).float()
        c = torch.tensor([self.c])
        scale = torch.tensor([self.scale])
        
        # create mesh data
        meshes = torch.tensor([])
        rotations = torch.tensor([])
        for idx, (image1, image2) in enumerate(zip(images1, images2)):
            image1, image2, offset, rotation = create_relative_postion(image1[:, :2], image2[:, :2], c, scale)
            mesh = mesh_from_image_meshes(image1, image2, offset)
            meshes = torch.cat((meshes, mesh.unsqueeze(0)), 0)
            rotations = torch.cat((rotations, rotation.unsqueeze(0)), 0)

        # display in blender    
        meshes = make_local_space(meshes)
        for mesh in meshes:
            create_object(mesh, f'mesh_{idx}')
        
        create_object(image1, 'image1')
        create_object(image2, 'image2')
        matrix = Matrix([
            [rotation[0][0], rotation[0][1], rotation[0][2], offset[0]],
            [rotation[1][0], rotation[1][1], rotation[1][2], offset[1]],
            [rotation[2][0], rotation[2][1], rotation[2][2], offset[2]],
            [0 ,0, 0, 1]])

        bpy.ops.pytorch.create_camera()
        cam = bpy.data.objects['camera_1']
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        cam.matrix_world = matrix

        return {'FINISHED'}


class pytorch_camera_position_auto(bpy.types.Operator):
    bl_idname = "pytorch.camera_position_auto"
    bl_label = "pytorch camera"
    bl_options = {'REGISTER','UNDO'}

    def execute(self, context):
        # create image data
        #dir1 = 'C://Users//macev//OneDrive//Documents//one drive//OneDrive//transfer files'
        dir1 = 'C://Users//Juan//OneDrive//transfer files'
        images1 = np.load(f'{dir1}/images_norm1.npy')
        images2 = np.load(f'{dir1}/images_norm2.npy')
        train(images1, images2)
        
        return {'FINISHED'}




class VIEW3D_PT_pytorch_camera(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'pytorch_camera'
    bl_label = 'pytorch_camera'
    
    def draw(self, context):     
        self.layout.label(text="pytorch camera")
        self.layout.operator('pytorch.camera_position_auto', text='find camera position')
