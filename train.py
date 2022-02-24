import bpy
import torch
import math
import numpy as np
from .epipole_utils import create_base_line
from .camera import camera_object
from .transform_utils import *
from .pytorch3d_utils import quaternion_to_matrix
import cv2


def norm_image(image, image_size):
    '''
    set the origin to image centre

    Args:
        image: tensor shape nx2
        image_size: int width and height of the image
    Return:
        tensor shape nx2
    '''
    middle = int(image_size / 2)
    image -= middle
    return image


def unnorm_image(image, image_size):
    '''
    set the origin to image top left

    Args:
        image: tensor shape nx2
        image_size: int, width and height of the image
    Return:
        tensor shape nx2
    '''
    middle = int(image_size / 2)
    image += middle
    return image


def show_points(points, render_size):
    '''
    takes an image and displays it

    Args:
        points: shape nx2 numpy or tensor
        render_size: int, width and height of the image
    '''
    if torch.is_tensor(points):
        points = points.clone().detach().numpy()


    print(points)
    image = np.zeros((render_size, render_size))
    for p in points:
        image = cv2.circle(image, p, 1, (255, 0, 0), 3)
    image = cv2.resize(image, (512, 512))
    cv2.imshow('test', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


def train():
    print('test')
    scale = torch.tensor(2.5 ,requires_grad=True).float()
    c = torch.tensor(-1. ,requires_grad=True).float()
    lr = 0.01  
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([c, scale], lr=lr)
    epochs = 200
    mesh_list = 14
    image_size = 1080

    # create target data
    target1 = np.load('C://Users//Juan//OneDrive//transfer files//target1.npy')
    target2 = np.load('C://Users//Juan//OneDrive//transfer files//target2.npy')
    target1 = torch.from_numpy(target1)
    target2 = torch.from_numpy(target2)

    pred_meshes = torch.tensor([])
    # create pred for each image
    for img_idx in range(target1.shape[0]):
        image1 = norm_image(target1[img_idx], image_size) / 1080
        image2 = norm_image(target2[img_idx], image_size) / 1080

        offset1, offset2 = create_base_line(
            make_homogeneous(image1), make_homogeneous(image2), c, scale)       
    
        # this creates the mesh local space 
        image_mesh1 = image_to_tensor(image1, c)
        image_mesh2 = image_to_tensor(image2, c)
        image_mesh1 = torch.vstack((image_mesh1, offset1.reshape((1, 3))))
        image_mesh2 = torch.vstack((image_mesh2, offset2.reshape((1, 3))))

        # placing img2 into world space assuming img1 is (0,0,0). do this by:
        # - offsetting img2 by img1 epipole
        # - rotating img2 so epipole lies along base line.
        offset_inverted = offset1 * -1
        quat = rotation_difference(image_mesh2[-1], offset_inverted.reshape(3))
        matrix_rot = quaternion_to_matrix(quat)
        image_mesh2 = transform_mesh_tensor(image_mesh2, matrix_rot)
        image_mesh2 = offset_verts(image_mesh2, offset_inverted.reshape(3))

        # - twist img2 along the base line so that tie points lie on the same plane.
        p1 = offset1.reshape(3) + image_mesh1[0]
        target = line_plane_intercept(image_mesh1[0], p1, image_mesh2[0], offset_inverted.reshape(3))
        target -= offset1.reshape(3)
        image_point = image_mesh2[0] - offset1.reshape(3)

        # replace with rotation difference but need to add custom axis angle option to it
        axis_vector = normalise_vector(offset1.reshape(3))
        angle = angle_between_two_vectors(image_point, target)
        quat = axis_angle_to_quaternion(angle, axis_vector)
        matrix_twist = quaternion_to_matrix(quat)

        #putting to origin and back so i can twist
        image_mesh2 = offset_verts(image_mesh2, offset_inverted.reshape(3))
        image_mesh2 = transform_mesh_tensor(image_mesh2, matrix_twist)
        image_mesh2 = offset_verts(image_mesh2, offset1.reshape(3))

        # create mesh
        mesh_3d = torch.tensor([])
        for vert1, vert2 in zip(image_mesh1, image_mesh2):
            origin1 = torch.tensor((0, 0, 0))
            direction1 = vert1
            origin2 = offset1.reshape(3)
            direction2 = vert2 + offset_inverted.reshape(3)
            middle = get_middle_point(origin1, direction1, origin2, direction2)
            mesh_3d = torch.cat((mesh_3d, middle.unsqueeze(0)), 0)
    
        pred_meshes = torch.cat((pred_meshes ,mesh_3d.unsqueeze(0)), 0)

    print(pred_meshes.shape)
    create_object(pred_meshes[0], '3d mesh')





