import bpy
import torch
from mathutils import Matrix
from .pytorch3d_utils import euler_angles_to_matrix


def loc_rot_to_matrix(location, rotation, convention):
    '''
    takes a location and rotation tensor and converts it to a 
    tensor that can translate and rotate vertices. 

    Args:
        location: 1d tensor of x, y, z coordinates
        rotation: 1d tensor of Euler angles in radians
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.
    Returns:
        transform matrix as tensor of shape (4, 4).
    '''
    # create 4x3 rotation 
    zero_row = torch.zeros(3)
    rotation = euler_angles_to_matrix(rotation, convention)
    rotation = torch.vstack((rotation, zero_row))

    # create 4x1 translation
    zero = torch.tensor((1.))
    # for some reason dim of 0 so changing it to 1
    zero = zero.reshape(1)
    location = torch.cat((location, zero))
    location = location.reshape((4, 1))

    # merge to 4x4
    transform = torch.hstack((rotation, location)) 
    return transform


def tensor_to_Matrix(tensor):
    '''
    converts a 4x4 tensor into a 4x4 mathutils.Matrix

    Args:
        tensor: a 4x4 tensor
    Returns:
        4x4 mathutils Matrix
    '''
    matrix = tensor.detach().tolist()
    matrix = Matrix(matrix)
    return matrix


def Matrix_to_tensor(Matrix):
    '''
    converts a 4x4 mathutils.Matrix into a 4x4 tensor

    Args:
        Matrix: a 4x4 mathutils Matrix
    Returns:
        4x4 tensor
    '''
    matrix_list = []
    for row in Matrix:
        row_list = []
        for i in row:
            row_list.append(i)
        matrix_list.append(row_list)

    matrix = torch.tensor(matrix_list)    
    return matrix
