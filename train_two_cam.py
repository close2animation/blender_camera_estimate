import bpy
import torch
import math
import numpy as np
from .transform_utils import *
from .pytorch3d_utils import quaternion_to_matrix
import cv2
from .blend_data_utils import *


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
    image = image - middle
    return image / image_size


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
    return image * image_size


def show_points(points, render_size):
    '''
    takes an image and displays it

    Args:
        points: shape nx2 numpy or tensor
        render_size: int, width and height of the image
    '''
    if torch.is_tensor(points):
        points = points.clone().detach().numpy()


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


def create_relative_postion(image1, image2, c, offset1, offset2, invert):
    '''
    takes an image and transforms it into a 3d object facing -z. y is hieght and x is width. 
    then rotates it so that image2's epipole lies on the same line as image1's epipole.
    then twists an arbitrary image point along the baseline till it lies on the epipolar plane. 
    Args:
        image1: 2d image points, tensor
        image2: 2d image points, tensor
        image_size: height and width of image, int. assumes square image.
        c: 1d tensor, camera constant

    Returns:
        image_mesh1: image1 as mesh with no rotation or location transform. 
        image_mesh2: image2 with transforms placing it relative to image1
        offset1: image2 offset from image1
    
    '''
    # this creates the mesh local space 
    print(c)
    offset1[2] = c
    offset2[2] = c
    image_mesh1 = image_to_tensor(image1, c)
    image_mesh2 = image_to_tensor(image2, c)
    # just for testing 
    #image_mesh1 = torch.vstack((image_mesh1, torch.tensor([0, 0, 0])))
    #image_mesh2 = torch.vstack((image_mesh2, torch.tensor([0, 0, 0])))
    image_mesh1 = torch.vstack((image_mesh1, offset1.reshape((1, 3))))
    image_mesh2 = torch.vstack((image_mesh2, offset2.reshape((1, 3))))

    # placing img2 into world space assuming img1 is (0,0,0). do this by:
    # - offsetting img2 by img1 epipole
    # - rotating img2 so epipole lies along base line.
    if invert:
        #offset_inverted = torch.tensor([(-offset1[0]), (-offset1[1]), offset1[2]])
        offset_inverted = offset1 * -1
    else:
        offset_inverted = offset1
    quat = rotation_difference(image_mesh2[-1], offset_inverted)
    matrix_rot = quaternion_to_matrix(quat)
    image_mesh2 = transform_mesh_tensor(image_mesh2, matrix_rot)
    image_mesh2 = offset_verts(image_mesh2, offset_inverted)

    # - twist img2 along the base line so that tie points lie on the same plane.
    p1 = offset1.reshape(3) + image_mesh1[0]
    target = line_plane_intercept(image_mesh1[0], p1, image_mesh2[0], offset_inverted)
    target -= offset1
    image_point = image_mesh2[0] - offset1

    # replace this with rotation difference but need to add custom axis angle option to it <-------------------------------- change this later
    axis_vector = normalise_vector(offset1)
    angle = angle_between_two_vectors(image_point, target)
    quat = axis_angle_to_quaternion(angle, axis_vector)
    matrix_twist = quaternion_to_matrix(quat)

    #putting to origin and back so i can twist
    image_mesh2 = offset_verts(image_mesh2, offset_inverted)
    image_mesh2 = transform_mesh_tensor(image_mesh2, matrix_twist)
    image_mesh2 = offset_verts(image_mesh2, offset1)
    return image_mesh1, image_mesh2, (matrix_twist @ matrix_rot)


def mesh_from_image_meshes(image_mesh1, image_mesh2, offset):
    '''
    projects a line from image point pairs and creates a vert at their closest point.
    '''
    offset_inverted = offset * -1
    mesh_3d = torch.tensor([])
    for vert1, vert2 in zip(image_mesh1, image_mesh2):
        origin1 = torch.tensor((0, 0, 0))
        direction2 = vert2 + offset_inverted
        middle = get_middle_point(origin1, vert1, offset, direction2)
        mesh_3d = torch.cat((mesh_3d, middle.unsqueeze(0)), 0)
    return mesh_3d


def create_preds(context, loc, rotation, c1, c2, images1, images2):
    '''
    takes the image data and creates a mesh with it. then reprojects said mesh back into image space. returns the results

    Args:
        context: bpy context
        loc: location of camera two. tensor shape[3]
        rotation: rotation of camera two. tensor shape[3]
        c: tensor shape[1]
        images1: tensor shape[n, 2]
        images2: tensor shape[n, 2]
    Returns:
        preds_1: image of newly created mesh in camera 1.tensor shape[]
        preds_2: image of newly created mesh in camera 2. tensor shape[]
    '''
    preds_1 = []
    preds_2 = []
    meshes = []
    for image1, image2 in zip(images1, images2):
        image1 = image_to_tensor(image1, c1)
        image2 = image_to_tensor(image2, c2)
        # adjust camera 2 
        rot = euler_angles_to_matrix(rotation, 'XYZ')
        image2 = transform_mesh_tensor(image2, rot)
        image2 = offset_verts(image2, -loc)
        mesh = mesh_from_image_meshes(image1, image2, loc)
        meshes.append(mesh)
        # create new image 1
        image1_pred = project_mesh_to_camera_space(make_homogeneous(mesh), c1)
        image1_pred = image1_pred[:, :2]
        preds_1.append(image1_pred)
        # create image 2
        mesh = offset_verts(mesh, loc)
        mesh = transform_mesh_tensor(mesh, rot.inverse())
        image2_pred = project_mesh_to_camera_space(make_homogeneous(mesh), c2)
        image2_pred = image2_pred[:, :2]
        preds_2.append(image2_pred)

    preds_1 = tensor_list_to_tensor(preds_1)    
    preds_2 = tensor_list_to_tensor(preds_2)    
    return preds_1, preds_2, meshes


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



