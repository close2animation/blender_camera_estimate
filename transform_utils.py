import bpy
import torch
from mathutils import Matrix, Vector
from .pytorch3d_utils import euler_angles_to_matrix
import bmesh


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
    matrix_list = [row[:] for row in Matrix]
    return torch.tensor(matrix_list)


def mesh_space_to_world(data, matrix_world):
    '''
    takes vertices in the mesh space and converts them to world space

    Args:
        data: bpy mesh type or list or verts.
        matrix_world: meshes 4x4 world matrix. mathutlis.Matrix
    Returns:
        list of mesh vertices in world space
    '''
    # checking if mesh type or list
    if isinstance(data, bpy.types.Object):
        if data.type != 'MESH':
            raise ValueError(f"Got an object of '{data.type}', but only 'MESH' type is supported.")
        verts = data.data.vertices
    elif isinstance(data, bpy.types.Mesh):
        verts = data.vertices
    elif isinstance(data, bmesh.types.BMesh):
        verts = data.verts
    elif isinstance(data, list):
        verts = None
    else:
        raise TypeError(f"Expected a 'MESH' object, a mesh, a bmesh or a list, got {data} instead.")
    
    if verts:
        return [matrix_world @ v.co for v in verts]
    else:
        return [matrix_world @ co for co in data]


def create_object(verts, object_name):
    '''
    creates an object and uses the verts list to define the mesh.

    Args:
        verts: nx3 list used to create mesh
        object_name: string that will become the objects name
    '''
    # create object
    mesh = bpy.data.meshes.new("mesh")  
    obj = bpy.data.objects.new(object_name, mesh)  

    # link to scene
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    
    # use bmesh to create object
    obj.select_set(True)
    mesh = bpy.context.object.data
    bm = bmesh.new()
    for v in verts:
        v = Vector(v)
        v.resize_3d()
        bm.verts.new(v) 

    # make the bmesh the object's mesh
    bm.to_mesh(mesh)
    bm.free()


def transform_mesh_tensor(mesh, transform):
    '''
    takes nx4 tensor and transforms it with a 4x4 transformation matrix.
    if the input is only shape nx3, a columns of 1's will be added.

    Args:
        mesh: nx4 tensor that describes a mesh
        transform: 4x4 tensor that describes a rotation and translation
    Returns:
        list of mesh vertices in world space
    '''
    if mesh.shape[1] == 3:
        ones = torch.ones((mesh.shape[0], 1))
        mesh = torch.hstack((mesh, ones))

    mesh_new = torch.tensor([])
    for vert in mesh:
        vert_new = torch.matmul(transform, vert)
        mesh_new = torch.cat((mesh_new, vert_new.unsqueeze(0)))
    return mesh_new


def vert_list_to_tensor(mesh):
    '''
    convert a nx3 list into a nx4 tensor. extra dim to make it homogeneous.

    Args:
        mesh: nx3 list that describes a mesh
    Returns:
        nx4 tensor that describes a mesh in homogeneous coordinates
    '''
    mesh = torch.tensor(mesh)
    dim = mesh.shape[0]
    ones = torch.ones((dim, 1))
    mesh = torch.hstack((mesh, ones))
    return mesh 

def tensor_to_vert_list(tensor):
    '''
    converts a nx4 tensor into a nx3 list

    Args:
        mesh: nx3 list that describes a mesh
    Returns:
        nx4 tensor that describes a mesh in homogeneous coordinates

    '''
    mesh = tensor.detach().tolist()
    for idx in range(tensor.shape[0]):
        mesh[idx][:] = [x / mesh[idx][3] for x in mesh[idx]]
        del mesh[idx][3]
    return mesh



