import torch
from .transform_utils import tensor_list

def calculate_fundamental_matrix(image2, image1):
    '''
    calculates the fundamental matrix which is used to create the epipoles.

    Args:
        image1: nx3 tensor of a mesh in the camera space 
        image2: nx3 tensor of a mesh in the another camera space
    Returns:
        fundamental matrix as 4x4 tensor
    '''
    A_list = [torch.kron(p1, p2) for p1, p2 in zip(image2, image1)]
    A = tensor_list_to_tensor(A_list)

    U,D,V = torch.linalg.svd(A)
    Fa = V[-1].reshape(3,3)

    # force rank 2
    U,D,V = torch.linalg.svd(Fa)
    D[2] = 0
    F = torch.matmul(U, torch.matmul(torch.diag(D), V))
    return F/F[2, 2]


def compute_epipole(F):
    '''
    computes the epipole which is a pixel in the camera space that reveals
    the direction of the other camera.

    Args:
        F: the fundamental matrix as a 4x4 tensor
    Returns:
        e: the epipole in camera space
    '''

    # return null space of F (Fx=0)
    U,D,V = torch.linalg.svd(F)
    e = V[-1]
    return e/e[2]


def calculate_epipolar_constraint(image1, image2):
    '''
    creates the epipolar line that the camera's location will
    be constrained too during training.

    Args:
        image1: an object projected into the camera space of camera 1.
        image2: the same object projected into the camera space of camera 2.
    Returns:
        tuple that contains the epipoles in camera 1 and camera 2's camera space

    '''
    F = calculate_fundamental_matrix(image2, image1)
    F_t = torch.transpose(F, 0, 1)
    e = [compute_epipole(F), compute_epipole(F_t)]
    return e

    
def tensor_list_to_tensor(tensor_list):
    '''
    turns a list of tensors into a tensor batch

    Args:
        list of tensors
    Returns:
        tensor
    '''
    # rewrite 151 camera
    tensor_batch = tensor_list[0].unsqueeze(0)
    for t in tensor_list[1:]:
        tensor_batch = torch.cat((tensor_batch, t.unsqueeze(0)), 0)
        
    return tensor_batch