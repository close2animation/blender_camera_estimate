import bpy
import torch
import math
from .epipole_utils import create_base_line
from .camera import camera_object
from .transform_utils import *


def create_transfroms(location, euler):
    '''
    creates a transform matrix from the location and rotation and applys it 
    to the camera object.

    Args:
        location: tuple of (x,y,z) location
        euler: tuple of (x,y,z) euler angles

    Returns:
        rotation and location matrix
    '''
    location = torch.tensor(location)
    euler = torch.tensor(euler)
    convention = 'XYZ'
    return loc_rot_to_matrix(location, euler, convention)


def create_image(cam, mesh=None):
    '''
    takes a mesh type or list of verts and creates 2d homogeneous points
    Args:
        cam: camera_object type
        mesh: mesh type or nx3 list 
    '''
    if mesh:
        mesh.mesh_norm = transform_mesh_tensor(mesh, torch.inverse(cam.matrix))
    else:
        cam.normalise_mesh()
    cam.project_mesh_to_camera_space()
    return cam.image_norm


def create_pred_mesh(offset1, direction_vectors1, offset2, direction_vectors2):
    '''
    creates the mesh we will project onto the camera planes.
    
    Args:
        offset1: camera 1 offset shape 3
        direction_vectors1: camera 1 image mesh, world space, shape nx3
        offset2: camera 2 offset shape 3
        direction_vectors2: camera 2 image mesh, world space, shape nx3 
    Returns:
        tensor mesh shape nx3
    '''
    pred_mesh = torch.tensor([])
    for dv1, dv2 in zip(direction_vectors1, direction_vectors2):
        middle_point = get_middle_point(offset1, 
                                        dv1,
                                        offset2, 
                                        dv2)
        pred_mesh = torch.cat((pred_mesh, middle_point.unsqueeze(0)), 0)
    return pred_mesh


def train_copy():
    print('test')

    x = math.radians(-45)
    y = math.radians(0)
    z = math.radians(0)
    scale = torch.tensor(2.48 ,requires_grad=True).float()
    c = torch.tensor(-1. ,requires_grad=True).float()
    rotation = torch.tensor((x, y, z), requires_grad=True).float()
    lr = 0.01  
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([c, scale, rotation], lr=lr)
    epochs = 150
    mesh_list = 14

    for epoch in range(epochs):
        loss_list = torch.tensor([])

        for mesh in range(mesh_list)[1:]:
            mesh = f'Icosphere_1.{mesh}'
            # create cameras
            cam1 = camera_object('camera_1')
            cam1.matrix = Matrix_to_tensor(cam1.bpy_camera.matrix_world)
            cam1.bpy_mesh = bpy.data.objects[mesh]
            cam1.constant = c
            
            cam2 = camera_object('camera_2')
            cam2.matrix = Matrix_to_tensor(cam2.bpy_camera.matrix_world)
            cam2.bpy_mesh = bpy.data.objects[mesh]
            cam2.constant = c

            # create target for testing. in real we'll use the mediapipe points here
            cam1.normalise_mesh()
            cam1.project_mesh_to_camera_space()
            cam2.normalise_mesh()
            cam2.project_mesh_to_camera_space()

            target1 = torch.tensor(cam1.image_norm)[:, :2]
            target2 = torch.tensor(cam2.image_norm)[:, :2]
            target = torch.vstack((target1, target2))

            offset = create_base_line(make_homogeneous(target1),
                                        make_homogeneous(target2), c, scale)

            offset_test = offset.reshape(3).detach().tolist()
            bpy.data.objects['Empty'].location = offset_test

            image_mesh = image_to_tensor(target1, c)
            image_mesh2 = image_to_tensor(target2, c)

            rot = loc_rot_to_matrix(torch.tensor([0, 0, 0]), rotation, 'XYZ')
            rot = rot[:3]
            rot = rot[:, :3]

            cam1.matrix = make_homogeneous(rot)
            direction_vectors = transform_mesh_tensor(image_mesh2, rot)
            pred_mesh = create_pred_mesh(torch.tensor([0, 0, 0]), 
                                        image_mesh,
                                        offset.reshape(3), 
                                        direction_vectors)

            cam1.mesh_norm = make_homogeneous(pred_mesh)
            pred_mesh = offset_verts(pred_mesh, offset)

            pred_mesh = transform_mesh_tensor(pred_mesh, torch.inverse(rot))
            cam2.mesh_norm = make_homogeneous(pred_mesh)

            cam1.project_mesh_to_camera_space()
            cam2.project_mesh_to_camera_space()

            preds = torch.vstack((cam1.image_norm, cam2.image_norm))
            loss = criterion(preds[:, :-1], target)
            #print('loss', loss)

            loss_list = torch.cat((loss_list, loss.unsqueeze(0)), 0)

        total_loss = torch.sum(loss_list) / loss_list.shape[0]
        print('total_loss')
        print(total_loss)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print('scale', scale)
    print('c', c)
    
    create_object(cam1.mesh_norm, 'result')
    #create_transfroms(base_line, rotation, cam2)
    #cam2.update_blend_camera
    





