import numpy as np
import torch
import skvideo.io
from mesh_viewer import MeshViewer
from human_body_prior.body_model.body_model import BodyModel
import trimesh
from trimesh.visual.texture import TextureVisuals
from PIL import Image
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import colors, apply_mesh_tranfsormations_
from math import pi

class NakedPeopleGenarator:
    def __init__(self, bmlmovi_path, smplh_path, dmpl_path):
        self.bmlmovi_path = bmlmovi_path
        self.smplh_path = smplh_path
        self.dmpl_path = dmpl_path

    def nakedgen(self, output_file, subject_id, pose_id, bg_image=None, bg_color='white', color='grey', rotation=0, imw=300, imh=300, frame_skip=2, scale=1, translation = [0, 0]):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        npz_bdata_path = "{}/Subject_{}_F_MoSh/Subject_{}_F_{}_poses.npz".format(self.bmlmovi_path, subject_id, subject_id, pose_id)
        bdata = np.load(npz_bdata_path)
        gender = bdata["gender"]
        bm_path = "{}/{}/model.npz".format(self.smplh_path, gender)
        dmpl_path = "{}/{}/model.npz".format(self.dmpl_path, gender)
        poses = torch.Tensor(bdata["poses"]).to(device)
        betas = torch.Tensor(bdata["betas"][:10][np.newaxis]).to(device)
        dmpls = torch.Tensor(bdata["dmpls"]).to(device)
        num_betas = 10
        num_dmpls = 8
        bm = BodyModel(bm_path=bm_path, num_betas=num_betas, num_dmpls=num_dmpls, path_dmpl=dmpl_path).to(device)
        wall_face = torch.IntTensor([[0, 1, 2, 3]]).to(device)
        faces = c2c(bm.f)
        mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
        writer = skvideo.io.FFmpegWriter(output_file,
                                    outputdict={"-vcodec":"libx264", "-pix_fmt": "yuv420p"})
        sq_scale = 2.1
        wall = torch.cat((
            torch.FloatTensor([[-sq_scale, sq_scale, -1]]).to(device),
            torch.FloatTensor([[-sq_scale, -sq_scale, -1]]).to(device),
            torch.FloatTensor([[sq_scale, -sq_scale, -1]]).to(device),
            torch.FloatTensor([[sq_scale, sq_scale, -1]]).to(device),
        )).to(device)
        uv = np.array([
            [0., 1],
            [0., 0],
            [1., 0],
            [1., 1.],
        ])
        if bg_image:
            im = Image.open(bg_image)
            texture = TextureVisuals(image=im, uv=uv)
        else:
            texture = None
        wall_mesh = trimesh.Trimesh(vertices=c2c(wall), faces=wall_face, visual=texture, vertex_colors=np.tile(colors[bg_color], (4, 1)))
        translation = np.array(translation)

        for fId in range(0, len(poses), frame_skip):
            root_orient = poses[fId:fId + 1, :3]
            pose_body = poses[fId:fId + 1, 3:66]
            pose_hand = poses[fId:fId + 1, 66:]
            dmpl = dmpls[fId:fId + 1]
            body = bm(pose_body=pose_body, pose_hand = pose_hand, betas=betas, root_orient=root_orient)
            body_mesh_wfingers = trimesh.Trimesh(vertices=c2c(body.v[0]), faces=faces, vertex_colors=np.tile(colors[color], (6890, 1)))
            apply_mesh_tranfsormations_([body_mesh_wfingers], trimesh.transformations.rotation_matrix(- pi / 1.9, (1, 0, 0)))
            apply_mesh_tranfsormations_([body_mesh_wfingers], trimesh.transformations.rotation_matrix(rotation, (0, 1, 0)))
            basepoint = body_mesh_wfingers.vertices[:, 2].max().item()
            measure = (body_mesh_wfingers.vertices[:, 1].max().item() - body_mesh_wfingers.vertices[:, 1].min().item())
            body_mesh_wfingers.vertices[:, 2] -= basepoint
            body_mesh_wfingers.vertices *= scale
            body_mesh_wfingers.vertices[:, 2] += basepoint
            body_mesh_wfingers.vertices[:, :2] += translation * measure
            mv.set_static_meshes([body_mesh_wfingers, wall_mesh])
            body_image_wfingers = mv.render(render_wireframe=False)
            writer.writeFrame(body_image_wfingers)
        writer.close()