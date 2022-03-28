import bpy
import torch
from .train_two_cam import *
from .blend_data_utils import *
from .epipole_utils import calculate_epipolar_constraint
import bpy
from mathutils import Matrix
from bpy.props import CollectionProperty, StringProperty
from bpy_extras.io_utils import ImportHelper 
import mediapipe as mp
from pathlib import Path


class My_settings(bpy.types.PropertyGroup):
    fps: bpy.props.FloatProperty(name="FPS", default=24.0, min=0, max=60)
    scale: bpy.props.FloatProperty(name="video scale", default=1, min=0, max=1)
    c1: bpy.props.FloatProperty(name="camera constant 1", default=-1, max=0)
    c2: bpy.props.FloatProperty(name="camera constant 2", default=-1, max=0)
    video_path: bpy.props.StringProperty(name="video path", default='') 
    lock_fps : bpy.props.BoolProperty(name='lock fps', default=False)
    t_hand : bpy.props.BoolProperty(name='track hand', default=True)
    t_head : bpy.props.BoolProperty(name='track head', default=True)
    t_pose : bpy.props.BoolProperty(name='track pose', default=True)
    cam_flip : bpy.props.BoolProperty(name='filp camera', default=False)
    video_idx : bpy.props.IntProperty(name='video index', default=1, min=1, max=10)
    tracking_confidence : bpy.props.FloatProperty(name="tracking confidence", default=.9, min=0, max=1)
    collection_1 : bpy.props.StringProperty(default='')
    collection_2 : bpy.props.StringProperty(default='')
    reduce_bool : bpy.props.BoolProperty(name='reduce images', default=True)
    reduce_amount : bpy.props.IntProperty(name='reduce amount', default=50, min=10)
    epochs : bpy.props.IntProperty(name='epochs', default=100, min=1)
    lr : bpy.props.FloatProperty(name="learning rate", default=0.1, max=1)
    loc_r : bpy.props.FloatVectorProperty(default=(0, 0, 0))
    find_c : bpy.props.BoolProperty(name='find c', default=False)


class OT_load_data(bpy.types.Operator, ImportHelper):  
    bl_idname = "load.data" 
    bl_label = "load data" 
    bl_options = {'UNDO'}
    directory : StringProperty(subtype='DIR_PATH')
    files : CollectionProperty(type=bpy.types.OperatorFileListElement)
    confidence = None

    def load_mp_tools(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose     
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=self.confidence, min_tracking_confidence=self.confidence, refine_landmarks=True) 
        self.hands = self.mp_hands.Hands(min_detection_confidence=self.confidence, min_tracking_confidence=self.confidence)
        self.pose = self.mp_pose.Pose(min_detection_confidence=self.confidence, min_tracking_confidence=self.confidence)


    def create_empty_for_every_landmark(self, name, total_landmarks):
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
        bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[name]    
               
        for idx in range(total_landmarks):
            bpy.ops.object.empty_add(type='PLAIN_AXES')
            obj = bpy.context.active_object
            obj.name = f'{name}_{idx}' 


    def save_landmarks_xyz_to_empties(self, landmarks, name, frame):
        try:
            for landmark_idx, landmark in enumerate(landmarks.landmark):
                empty_name = name + '_' + str(landmark_idx)   
                obj =  bpy.data.objects[empty_name]
                obj.location[0] = landmark.x
                obj.location[1] = landmark.y
                obj.location[2] = landmark.z
                obj.keyframe_insert(data_path="location", frame=frame)
        except:
            pass
                  
    
    def loop_through(self, operation, landmark_list, name, frame):
        '''
        need to loop through face and hands since more than face/hand can exist at 1 time. 
        also need to check if landsmarks exist.
        '''  
        if not landmark_list:
            #print(f'no results in frame{frame} for {name}')
            pass
        else:
            try:
                for idx, landmarks in enumerate(landmark_list):
                    #print(idx)
                    operation(landmarks, name[idx], frame)
            except IndexError:
                pass

    
    def create_camera(self, video_path):
        # switch to object mode 
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode='OBJECT')  

        bpy.ops.object.camera_add()
        scene = bpy.context.scene
        scene.render.resolution_x = 1000
        scene.render.resolution_y = 1000
        camera = bpy.context.active_object
        camera.name = 'visualise points'
        camera.location[0] = .5
        camera.location[1] = .5
        camera.location[2] = -1
        camera.rotation_euler[0] = math.radians(180)
        camera.rotation_euler[1] = 0
        camera.rotation_euler[2] = 0
        camera.data.type = 'ORTHO'
        camera.data.ortho_scale = 1.0
        camera.data.show_background_images = True

        video = bpy.data.movieclips.load(video_path)
        bg = camera.data.background_images.new()
        bg.source = 'MOVIE_CLIP'
        bg.clip = video
        bg.rotation = math.radians(180)
        bg.use_flip_y = True
            
    
    def execute(self, context):
        # lets user find video path and returns it as string
        base = Path(self.directory)                
        for f in self.files:
            video_path = base / f.name
            print(video_path)
        
        video_path = str(video_path)
        my_tool = context.scene.my_tool
        my_tool.video_path = video_path
        self.confidence = my_tool.tracking_confidence

        # need to change this to: if not video_path.split('.')[0] in list_with_valid_video_formats: pick correct format
        if not video_path.split('.')[-1] == 'mp4':
            print('select a mp4 file')
        
        else:
            #params
            cap = cv2.VideoCapture(my_tool.video_path)
            scale = my_tool.scale
            frame = 0
            frame_rate= my_tool.fps
            count = 0
            lock_fps = my_tool.lock_fps
            hand_side = [f'{my_tool.video_idx}_hand_l', f'{my_tool.video_idx}_hand_r']
            t_hand = my_tool.t_hand
            t_head = my_tool.t_head
            t_pose = my_tool.t_pose
            self.load_mp_tools()
            self.create_camera(video_path)

            if t_head:
                print('creating empties for head')
                self.create_empty_for_every_landmark(f'{my_tool.video_idx}_face', 478)
            if t_hand:
                print('creating empties for hands')
                self.create_empty_for_every_landmark(hand_side[0], 21)
                self.create_empty_for_every_landmark(hand_side[1], 21)
            if t_pose:
                print('creating empties for body')
                self.create_empty_for_every_landmark(f'{my_tool.video_idx}_pose', 33)
                        
            while cap.isOpened():           
                if lock_fps:
                    cap.set(cv2.CAP_PROP_POS_MSEC,(count*1000)) # doing this takes twice as long.....
                    count += (1/frame_rate)      
                success, img = cap.read()
                if not success:
                    break
                print(f'processing frame:{frame}') 
                # process img
                img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
                #img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img.flags.writeable = False

                face_results = None
                hand_results = None
                pose_results = None
                if t_head:
                    face_results = self.face_mesh.process(img)
                    self.loop_through(self.save_landmarks_xyz_to_empties, face_results.multi_face_landmarks, [f'{my_tool.video_idx}_face'], frame)
                if t_hand:
                    hand_results = self.hands.process(img)
                    self.loop_through(self.save_landmarks_xyz_to_empties, hand_results.multi_hand_landmarks, hand_side, frame) 
                if t_pose:
                    pose_results = self.pose.process(img)
                    if pose_results.pose_landmarks:
                        self.save_landmarks_xyz_to_empties(pose_results.pose_landmarks, f'{my_tool.video_idx}_pose', frame) 
                
                frame += 1     
            cv2.destroyAllWindows()
        return {'FINISHED'}


class VIEW3D_PT_load_data(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'pytorch_camera'
    bl_label = 'pytorch_camera'
    
    def draw(self, context):     
        self.layout.label(text="processing options")
        my_tool = context.scene.my_tool
        
        row = self.layout.row()
        row.prop(my_tool, "tracking_confidence")
        #row.prop(my_tool, "lock_fps")
        row.prop(my_tool, "scale")
        
        row = self.layout.row()
        row.prop(my_tool, "t_head")
        row.prop(my_tool, "t_hand")
        row.prop(my_tool, "t_pose")

        row = self.layout.row()
        row.prop(my_tool, "video_idx")
        row.operator('load.data', text='select mp4 file')


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


class pytorch_camera_save_epipole(bpy.types.Operator):
    bl_idname = "pytorch.camera_save_epipole"
    bl_label = "pytorch camera"
    bl_options = {'REGISTER','UNDO'}

    def execute(self, context):
        my_tool = context.scene.my_tool
        images1, images2 = collect_images(my_tool.collection_1, my_tool.collection_2)
        if not images1.shape == images2.shape:
            self.report({"WARNING"}, "image points don't match")
            return {"CANCELLED"}
        # transform to single image
        dim = (images1.shape[0] * images1.shape[1])
        image1 = images1.reshape((dim, 2))
        dim = (images2.shape[0] * images2.shape[1])
        image2 = images2.reshape((dim, 2))

        print(images1.shape)
        print(images2.shape)
        epipole = calculate_epipolar_constraint(make_homogeneous(image2), make_homogeneous(image1))
        #epipole[1], epipole[0] = epipole[0], epipole[1] 
        for idx, e in enumerate(epipole):
            # might need to move this
            e[2] = -e[2]
            bpy.ops.object.empty_add(type='PLAIN_AXES')
            bpy.context.active_object.name = f'epipole {2 - idx}'
            bpy.context.active_object.location = e.numpy()
        return {'FINISHED'}


class pytorch_camera_position_pass1(bpy.types.Operator):
    bl_idname = "pytorch.camera_position_pass1"
    bl_label = "pytorch camera"
    bl_options = {'REGISTER','UNDO'}

    c: bpy.props.FloatProperty(
        name="constant",
        default=-2.6667)

    def execute(self, context):
        my_tool = context.scene.my_tool
        c = torch.tensor([self.c])
        e1 = np.array(bpy.data.objects['epipole 1'].location)
        e1 = torch.from_numpy(e1).float()
        e2 = np.array(bpy.data.objects['epipole 2'].location)
        e2 = torch.from_numpy(e2).float()

        # create image data
        images1, images2 = collect_images(my_tool.collection_1, my_tool.collection_2)
        images1, images2 = reduce_img_amount(images1, images2, 6)

        # create mesh data
        meshes = torch.tensor([])
        rotations = torch.tensor([])
        for idx, (image1, image2) in enumerate(zip(images2, images1)):
            image1, image2, rotation = create_relative_postion(image1, image2, c, e1, e2, my_tool.cam_flip)
            mesh = mesh_from_image_meshes(image1, image2, e1)
            meshes = torch.cat((meshes, mesh.unsqueeze(0)), 0)
            rotations = torch.cat((rotations, rotation.unsqueeze(0)), 0)

        '''
        # display in blender    
        meshes = make_local_space(meshes)
        for idx, mesh in enumerate(meshes):
            create_object(mesh, f'mesh_{idx}')
        '''
        #create_object(image1, 'image1')
        #create_object(image2, 'image2')
        print('av dim', rotations.mean(0))
        rotation = rotations.mean(0)
        matrix = Matrix([
            [rotation[0][0], rotation[0][1], rotation[0][2], e1[0]],
            [rotation[1][0], rotation[1][1], rotation[1][2], e1[1]],
            [rotation[2][0], rotation[2][1], rotation[2][2], c],
            [0 ,0, 0, 1]])

        bpy.ops.pytorch.create_camera()
        cam = bpy.data.objects['camera_1']
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        cam.matrix_world = matrix
        return {'FINISHED'}


class pytorch_camera_position_pass2(bpy.types.Operator):
    bl_idname = "pytorch.camera_position_pass2"
    bl_label = "pytorch camera"
    bl_options = {'REGISTER','UNDO'}

    def execute(self, context):
        my_tool = context.scene.my_tool
        criterion = torch.nn.MSELoss()
        c1 = torch.tensor([my_tool.c1], requires_grad=True)
        c2 = torch.tensor([my_tool.c2], requires_grad=True)
        loc = bpy.data.objects['camera_2'].location
        loc = torch.tensor(loc)
        loc_r = torch.tensor(my_tool.loc_r, requires_grad=True)
        rot = bpy.data.objects['camera_2'].rotation_euler
        rot = torch.tensor(rot, requires_grad=True)

        if my_tool.find_c:
            optimizer = torch.optim.Adam([c1, c2, loc_r, rot], lr=my_tool.lr)
        else:
            optimizer = torch.optim.Adam([loc_r, rot], lr=my_tool.lr)

        # create image data
        images1, images2 = collect_images(my_tool.collection_1, my_tool.collection_2)
        if my_tool.reduce_bool:
            images1, images2 = reduce_img_amount(images1, images2, my_tool.reduce_amount)
        images1 = tensor_list_to_tensor(images1)
        images2 = tensor_list_to_tensor(images2)

        if not images1.shape == images2.shape:
            self.report({"WARNING"}, "image points don't match")
            return {"CANCELLED"}

        for epoch in range(my_tool.epochs):
            # images loss
            loc_new = euler_angles_to_matrix(loc_r, 'XYZ')
            loc_new = loc_new @ loc
            images1_pred, images2_pred, meshes = create_preds(context, loc_new, rot, c1, c2, images1, images2)

            loss1 = criterion(images1_pred, images1)
            loss2 = criterion(images2_pred, images2)
            loss = loss1 + loss2
            print('epoch: ', epoch, f'{(epoch/my_tool.epochs) * 100}% done.')
            print('rotation: ', rot)
            print('location: ', loc_r)
            print('c1: ', c1)
            print('c2: ', c2)
            print('loss: ', loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        my_tool.lr = get_lr(optimizer)
        print(my_tool.lr)
        # update camera
        bpy.data.objects['camera_2'].location = (loc_new[0], loc_new[1], loc_new[2])
        bpy.data.objects['camera_2'].rotation_euler = (rot[0], rot[1], rot[2])
        #vis_results('hand_bone_1_none', 'bone_joints5', meshes)
        meshes = tensor_list_to_tensor(meshes)
        meshes = meshes.detach().clone().numpy()
        create_object(meshes[0], 'results')
        create_mesh_animation(meshes, 'results')
        my_tool.c1 = c1.detach().clone().numpy()
        my_tool.c2 = c2.detach().clone().numpy()
        my_tool.loc_r = loc_r.detach().clone().numpy()
        return {'FINISHED'}


class pytorch_create_animation(bpy.types.Operator):
    bl_idname = "pytorch.create_animation"
    bl_label = "pytorch camera"
    bl_options = {'REGISTER','UNDO'}

    def execute(self, context):
        my_tool = context.scene.my_tool
        c = torch.tensor([my_tool.c])


        return {'FINISHED'}


class VIEW3D_PT_pytorch_camera(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'pytorch_camera'
    bl_label = 'pytorch_camera'
    
    def draw(self, context):
        my_tool = context.scene.my_tool     
        self.layout.label(text="pytorch camera")
        row = self.layout.row()
        row.prop(my_tool, "collection_1")
        row.prop(my_tool, "collection_2")
        row = self.layout.row()
        row.prop(my_tool, "reduce_bool")
        row.prop(my_tool, "reduce_amount")
        row.prop(my_tool, "epochs")
        row = self.layout.row()
        row.prop(my_tool, "lr")
        row.prop(my_tool, "c1")
        row.prop(my_tool, "c2")
        row.prop(my_tool, 'find_c')
        row = self.layout.row()
        #self.layout.operator('pytorch.camera_save_epipole', text='save epipole')
        #self.layout.operator('pytorch.camera_position_pass1', text='camera position first pass')
        self.layout.operator('pytorch.camera_position_pass2', text='find camera position')
        #self.layout.operator('pytorch.create_animation', text='create animation')

       


