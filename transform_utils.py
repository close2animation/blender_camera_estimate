import bpy
import torch
from mathutils import Matrix, Vector
from .pytorch3d_utils import euler_angles_to_matrix
import bmesh
from .pytorch3d_utils import quaternion_to_matrix


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
    # in case of 3x1
    location = location.reshape(3)
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
    if torch.is_tensor(verts):
        verts = verts.clone().detach().tolist()

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
    takes shape(n,x) tensor and transforms it with a shape(x,x) transformation matrix.

    Args:
        mesh: shape(n,x) tensor that describes a mesh
        transform: shape(x,x) tensor that describes a rotation and translation
    Returns:
        list of mesh vertices in world space
    '''
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


def image_to_tensor(img, c):
    '''
    adds a z axis to an image and makes it a homogeneous tensor.

    Args:
        image: nx2 2d list
        c: tensor size 1. camera constant to get correct pixel normals
    Returns:
        image as a nx4 tensor
    '''
    img = torch.tensor(img)

    # add z axis
    dim = img.shape[0]
    c_list = [c for i in range(dim)]
    c = tensor_list_to_tensor(c_list)
    c = c.reshape((dim, 1))
    img = torch.hstack((img, c))
    return img

    
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


def make_homogeneous(tensor):
    '''
    adds a row of ones to a tensor to make it homogeneous

    Args:
        tensor: a nxn tensor
    Returns:
        tensor: a nx(n+1) tensor
    ''' 
    dim = tensor.shape[0]
    ones = torch.ones((dim,1))
    tensor = torch.hstack((tensor, ones))
    return tensor


def remove_homogeneous(tensor):
    '''
    divides tensor by last row then removes it. this is broken

    Args:
        tensor: tensor shape nxn
    Returns
        tensor: tensor without last column, shape nx(n-1)
    '''
    new_tensor = torch.tensor([])
    for i in range(tensor.shape[1]):
        t = tensor[:, i] / tensor[:, -1]
        t = t.reshape((tensor.shape[0], 1))
        new_tensor = torch.cat((new_tensor, t), 0)
    
    return tensor[:, :-1]



def get_middle_point(l1, u1, l2, u2):
    '''
    returns the closest point between two skew lines
    
    Args:
        l1: origin of line 1
        u1: direction of line 1
        l2: origin of line 2
        u2: direction of line 2
    Returns:
        shape [3] tensor that describes the middle point 
    '''
    l = l2 -l1
    PQ = torch.cat((u2.unsqueeze(0),
                    -u1.unsqueeze(0),
                    l.unsqueeze(0)), 0)
    PQ = torch.transpose(PQ, 0, 1)

    A1 = PQ[0] * u1[0]
    A2 = PQ[1] * u1[1]
    A3 = PQ[2] * u1[2]
    Ap = A1 + A2 + A3

    A1 = PQ[0] * u2[0]
    A2 = PQ[1] * u2[1]
    A3 = PQ[2] * u2[2]
    Aq = A1 + A2 + A3

    b = -torch.cat((Ap[2].unsqueeze(0), Aq[2].unsqueeze(0)), 0)              
    A = torch.cat((Ap[:2].unsqueeze(0), Aq[:2].unsqueeze(0)), 0)

    x = A.pinverse() @ b
    #x = torch.linalg.lstsq(A, b)[0]

    p = u1 * x[1] + l1
    q = u2 * x[0] + l2  
    middle_point = (p + q) / 2
    return middle_point   


def offset_verts(mesh, offset):
    '''
    this is super inconsistent will need to redo training loop
    '''
    new_mesh = [vert - offset.reshape(3) for vert in mesh]
    new_mesh = tensor_list_to_tensor(new_mesh)
    return new_mesh
        

def location_to_matrix(location):
    '''
    creates a 4x4 matrix that can be used to translate
    Args:
        location: tensor shape[3] 
    '''
    I = torch.tensor([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
    location = torch.vstack((location, torch.tensor(0)))
    location = torch.hstack((I, location.reshape(4,1)))
    return location


def normalise_vector(vector):
    '''
    normalising vectors so that (a**2 + b**2 + c**2 == 1**2) 

    Args:
        vector: shape[3] vector
    Returns:
        shape[3] vector with square sum equaling 1
    '''
    vector_s = vector ** 2
    vector_s = torch.sum(vector_s)
    vector_s = torch.sqrt(vector_s)
    return vector / vector_s


def axis_angle_to_quaternion(angle, vector):
    '''
    converts axis_angle (theta, vector) to quaternion (w, x, y, z). 
    formula for defining (w, x, y, z) <-- (cos(theta/2), v1*sin(theta/2), v2*sin(theta/2), v3*sin(theta/2)) 
    Args:
        angle: shape[1] angle to rotate
        vector: shape[3] axis angle
    Returns:
        shape[4] quat
    '''
    w = torch.cos(angle/2)
    xyz = torch.sin(angle/2) * vector
    quat = torch.hstack((w,xyz))
    return quat


def angle_between_two_vectors(vec1, vec2):
    '''
    takes two vectors with a magnitude of 1 and finds the angle between them

    Args:
        vec1: tensor shape[3]
        vec2: tensor shape[3]
    Returns:
        angle float tensor
    '''
    vec1 = normalise_vector(vec1)
    vec2 = normalise_vector(vec2)
    dot = torch.dot(vec1, vec2)
    # this might error
    angle = torch.arccos(dot/1)
    return angle


def rotation_difference(point, target):
    '''
    finds the rotation difference between two vectors in quaternions
    Args:
        point: shape[3] tensor defining first point
        target: shape[3] tensor defining second point rotation is towards
    Returns:
        shape[4] quat that will rotate point to target
    '''
    axis_vector = torch.cross(point, target)
    #print('axis_vector', axis_vector)
    axis_vector = normalise_vector(axis_vector)
    angle = angle_between_two_vectors(point, target)
    #print('angle: ', torch.rad2deg(angle))
    quat = axis_angle_to_quaternion(angle, axis_vector)
    '''
    target = target.reshape(3).detach().tolist()
    bpy.data.objects['target'].location = target

    point = point.reshape(3).detach().tolist()
    bpy.data.objects['point'].location = point

    axis_vector = axis_vector.reshape(3).detach().tolist()
    bpy.data.objects['axis_vector'].location = axis_vector
    '''
    return quat


def line_plane_intercept(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    p0, p1: Define the line.
    p_co, p_no: define the plane:
        p_co Is a point on the plane (plane coordinate).
        p_no Is a normal vector defining the plane direction;
             (does not need to be normalized).

    Return a Vector or None (when the intersection can't be found).
    """
    epsilon = torch.tensor([epsilon])  
    u = p1 - p0
    dot = torch.dot(p_no, u) 
    
    if torch.abs(dot) > epsilon:
        w = p0 - p_co
        fac = -torch.dot(p_no, w) / dot
        u = u * fac
        return p0 + u
    return None


def make_local_space(meshes):
    meshes_new = torch.tensor([])
    for i, mesh in enumerate(meshes):
    #mesh = meshes[-3]
        origin = mesh[0]
        mesh = offset_verts(mesh, origin)

        quat = rotation_difference(mesh[-1], torch.tensor((0, 0, 1.)))
        mat_rot = quaternion_to_matrix(quat)
        mesh = transform_mesh_tensor(mesh, mat_rot)

        target = torch.tensor((0, 1., 0))
        point = torch.tensor((mesh[5][0], mesh[5][1], 0))
        quat = rotation_difference(point, target)
        mat_rot = quaternion_to_matrix(quat)
        mesh = transform_mesh_tensor(mesh, mat_rot)
        meshes_new = torch.cat((meshes_new, mesh.unsqueeze(0)), 0)

    return meshes_new 


def save_angles(meshes):
    mesh_list = torch.tensor([])
    for mesh in meshes:
        angle_list = torch.tensor([])
        for vert in mesh[1:]:
            angle = angle_between_two_vectors(vert, mesh[11])
            angle_list = torch.cat((angle_list, angle.unsqueeze(0)), 0)
        mesh_list = torch.cat((mesh_list, angle_list.unsqueeze(0)), 0)
    return mesh_list
