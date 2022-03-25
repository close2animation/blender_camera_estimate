import bpy
import numpy as np
import torch
from .transform_utils import tensor_list_to_tensor

def get_specific_fcurves(obj, transform_type, axis):
    '''
    take an obj and returns a specific transform
    '''
    axes = {'x' : [0], 'y' : [1], 'z' : [2], 'all' : [0, 1, 2]}
    list = []
        
    for fcurve in obj.animation_data.action.fcurves:
        if fcurve.data_path == transform_type and fcurve.array_index in axes[axis]:
            list.append(fcurve)
    return list


def fcurve_to_list(fcurve):
    keypoints = []
    for frame in fcurve.keyframe_points:
        keypoints.append(frame.co[1])
    return keypoints
      
              
def collect_data_in_collection(name):
    data = []
    for obj in bpy.data.collections[name].objects:
        fcurves = get_specific_fcurves(obj, 'location', 'all')
        location = []
        for fcurve in fcurves:
            location.append(fcurve_to_list(fcurve))
        data.append(location)
    return data


def move_to_world_origin(obj, idx):
    '''
    takes a list of objects and slides the choosen vert to 0,0,0
    '''
    obj -= np.reshape(obj[:, idx], (obj.shape[0], 1, obj.shape[2]))
    return obj


def norm_image(data):
    # flip
    data[:, :, 1] = 1 - data[:, :, 1]
    # centre
    data -= .5
    data = data[:, :, :2]
    # transform to single image
    #dim = (data.shape[0] * data.shape[1])
    #data = data.reshape((dim, 2))
    return data


def collect_images(collection_1, collection_2):
    # create image data
    images1 = collect_data_in_collection(collection_1)
    images2 = collect_data_in_collection(collection_2)
    # set data shape to frames/obj/xyz
    images1 = np.transpose(np.array(images1), (2, 0, 1))
    images2 = np.transpose(np.array(images2), (2, 0, 1))
    images1 = norm_image(images1)
    images2 = norm_image(images2)
    images1 = torch.from_numpy(images1).float()
    images2 = torch.from_numpy(images2).float()
    return images1, images2


def reduce_img_amount(images1, images2, split):
    split_amount =  int(images1.shape[0] / (split - 1))
    spilts = [(split_amount * i) for i in range(split)]
    spilts[-1] = spilts[-1] - 1
    images1_reduced = [images1[idx] for idx in spilts]
    images2_reduced = [images2[idx] for idx in spilts]
    return images1_reduced, images2_reduced


def create_mesh_animation(animation_data, object_name):
    ''' input shape ---> frames/vert/coor '''
    animation_data = np.transpose(animation_data, (1, 0, 2))
    obj = bpy.data.objects[object_name]
    mesh = obj.data
    action = bpy.data.actions.new("meshAnimation")
    mesh.animation_data_create()
    mesh.animation_data.action = action

    # loop over verts
    for idx, vert in enumerate(animation_data):
        # create fcurves for vert (xyz)
        fcurves = [action.fcurves.new(f'vertices[{idx}].co', index=i) for i in range(3)]  
        for frame, frame_data in enumerate(vert):    
            fcurves[0].keyframe_points.insert(frame, frame_data[0], options={'FAST'}) # x
            fcurves[1].keyframe_points.insert(frame, frame_data[1], options={'FAST'}) # y
            fcurves[2].keyframe_points.insert(frame, frame_data[2], options={'FAST'}) # z


def create_hand(mesh, arm_name):
    bpy.ops.object.armature_add()
    obj = bpy.context.active_object
    obj.name = arm_name
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    obj.data.edit_bones.remove(obj.data.edit_bones[0])

    names = ['thumb', 'finger1', 'finger2', 'finger3', 'finger4']
    for idx, name in enumerate(names):
        idx = ((idx + 1) * 4) + 1 
        base = mesh[0]
        for i, joint_loc in enumerate(mesh[(idx - 4):idx]):
            joint = obj.data.edit_bones.new(f'{name}_{i}')
            joint.head = base
            joint.tail = joint_loc
            base = joint_loc
          
    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)


def create_emptys(name, total_emptys):
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[name]    
            
    for idx in range(total_emptys):
        bpy.ops.object.empty_add(type='PLAIN_AXES')
        obj = bpy.context.active_object
        obj.name = f'{name}_{idx}' 


def save_xyz_to_empties(coordinates, name, frame):
    for idx, co in enumerate(coordinates):
        empty_name = f'{name}_{idx}'   
        obj =  bpy.data.objects[empty_name]
        obj.location = co
        obj.keyframe_insert(data_path="location", frame=frame)


def animate_hand(object_name, empty_names):
    arm = bpy.data.objects[object_name]
    pb = arm.pose.bones
    
    for idx in range(5):
        idx = ((idx + 1) * 4)
        base = bpy.data.objects[f'{empty_names}_0']   
        for i, bone in enumerate(pb[(idx - 4):idx]):    
            constraint = bone.constraints.new('COPY_LOCATION')
            constraint.target = base
            base = bpy.data.objects[f'{empty_names}_{(i + (idx - 4)) + 1}']

    for idx, bone in enumerate(pb):
        empty = bpy.data.objects[f'{empty_names}_{idx + 1}']
        constraint = bone.constraints.new('STRETCH_TO')
        constraint.target = empty


def vis_results(arm_name, empty_name, meshes):
    meshes = tensor_list_to_tensor(meshes)
    mesh = meshes[-1].detach().clone().numpy()
    create_hand(mesh, arm_name)
    create_emptys(empty_name, 21)
    meshes = meshes.detach().clone().numpy()
    for frame, mesh in enumerate(meshes):
        save_xyz_to_empties(mesh, empty_name, frame)
    bpy.context.view_layer.update()
    animate_hand(arm_name, empty_name)
