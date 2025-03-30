import torch
import numpy as np
import argparse
import pickle
import smplx

from utils import bvh, quat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="data/smpl/")
    parser.add_argument("--model_type", type=str, default="smplx", choices=["smpl", "smplx"])
    parser.add_argument("--gender", type=str, default="NEUTRAL", choices=["MALE", "FEMALE", "NEUTRAL"])
    parser.add_argument("--num_betas", type=int, default=300, choices=[10, 300])
    parser.add_argument("--poses", type=str, default="data/gWA_sFM_cAll_d27_mWA5_ch20.pkl")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=str, default="data/gWA_sFM_cAll_d27_mWA5_ch20.bvh")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--order", type=str, default="zxy", help="The Euler angle rotation order in the bvh file.")
    return parser.parse_args()

def mirror_rot_trans(lrot, trans, names, parents):
    joints_mirror = np.array([(
        names.index("Left"+n[5:]) if n.startswith("Right") else (
        names.index("Right"+n[4:]) if n.startswith("Left") else 
        names.index(n))) for n in names])

    mirror_pos = np.array([-1, 1, 1])
    mirror_rot = np.array([1, 1, -1, -1])
    grot = quat.fk_rot(lrot, parents)
    trans_mirror = mirror_pos * trans
    grot_mirror = mirror_rot * grot[:,joints_mirror]
    
    return quat.ik_rot(grot_mirror, parents), trans_mirror

def smpl2bvh(model_path:str, poses:str, output:str, mirror:bool,
             order:str="zxy", model_type="smplx", gender="MALE",
             num_betas=10, fps=60) -> None:
    """Save bvh file created by smpl parameters.

    Args:
        model_path (str): Path to smpl models.
        poses (str): Path to npz or pkl file.
        output (str): Where to save bvh.
        mirror (bool): Whether save mirror motion or not.
        model_type (str, optional): I prepared "smpl" only. Defaults to "smpl".
        gender (str, optional): Gender Information. Defaults to "MALE".
        num_betas (int, optional): How many pca parameters to use in SMPL. Defaults to 10.
        fps (int, optional): Frame per second. Defaults to 30.
    """
    
    names = {
        "smpl": [
            "Pelvis",
            "Left_hip",
            "Right_hip",
            "Spine1",
            "Left_knee",
            "Right_knee",
            "Spine2",
            "Left_ankle",
            "Right_ankle",
            "Spine3",
            "Left_foot",
            "Right_foot",
            "Neck",
            "Left_collar",
            "Right_collar",
            "Head",
            "Left_shoulder",
            "Right_shoulder",
            "Left_elbow",
            "Right_elbow",
            "Left_wrist",
            "Right_wrist",
            "Left_palm",
            "Right_palm",
        ],
        "smplx": [
            "root",
            "pelvis", # 0, -1
            "left_hip", #1, 0
            "right_hip", #2, 0
            "spine1", #3, 0
            "left_knee", #4, 1
            "right_knee", #5, 2
            "spine2", #6, 3
            "left_ankle", #7, 4
            "right_ankle", #8, 5
            "spine3", #9, 6
            "left_foot", #10, 7
            "right_foot", #11, 8
            "neck", #12, 9
            "left_collar", #13, 9
            "right_collar", #14, 9
            "head", #15, 12
            "left_shoulder", #16, 13
            "right_shoulder", #17, 14
            "left_elbow", #18, 16
            "right_elbow", #19, 17
            "left_wrist", #20, 18
            "right_wrist", #21, 19
            "jaw", #22, 15
            "left_eye_smplhf", #23, 15
            "right_eye_smplhf", #24, 15
            "left_index1", #25, 20
            "left_index2", #26, 25
            "left_index3", #27, 26
            "left_middle1", #28, 20
            "left_middle2", #29, 28
            "left_middle3", #30, 29
            "left_pinky1", #31, 20
            "left_pinky2", #32, 31
            "left_pinky3", #33, 32
            "left_ring1", #34, 20
            "left_ring2", #35, 34
            "left_ring3", #36, 35
            "left_thumb1", #37, 20
            "left_thumb2", #38, 37
            "left_thumb3", #39, 38
            "right_index1", #40, 21
            "right_index2", #41, 40
            "right_index3", #42, 41
            "right_middle1", #43, 21
            "right_middle2", #44, 43
            "right_middle3", #45, 44
            "right_pinky1", #46, 21
            "right_pinky2", #47, 46
            "right_pinky3", #48, 47
            "right_ring1", #49, 21
            "right_ring2", #50, 49
            "right_ring3", #51, 50
            "right_thumb1", #52, 21
            "right_thumb2", #53, 52
            "right_thumb3", #54, 53
        ]
    }
    
    # I prepared smpl models only, 
    # but I will release for smplx models recently.
    model = smplx.create(model_path=model_path, 
                        model_type=model_type,
                        gender=gender, 
                        batch_size=1)
    
    parents = model.parents.detach().cpu().numpy()
    
    # You can define betas like this.(default betas are 0 at all.)
    rest = model(
        # betas = torch.randn([1, num_betas], dtype=torch.float32)
    )
    rest_pose = rest.joints.detach().cpu().numpy().squeeze()

    if model_type == "smpl":
        rest_pose = rest_pose[:24,:]
    elif model_type == "smplx":
        rest_pose = rest_pose[:55,:]
    else:
        raise ValueError("This model type is not supported!")
    
    root_offset = rest_pose[0]
    offsets = rest_pose - rest_pose[parents]
    offsets[0] = root_offset
    offsets *= 100

    scaling = None
    
    # Pose setting.
    if poses.endswith(".npz"):
        poses = np.load(poses)

        if poses["poses"].ndim == 2:
            rots = poses["poses"].reshape((-1, 55, 3)) # (N, 55 3)
        else:
            rots = np.squeeze(poses["poses"], axis=0) # (N, 24, 3)

        if poses["trans"].ndim == 2:
            trans = poses["trans"] # (N, 3)
        else:
            trans = np.squeeze(poses["trans"], axis=0) # (N, 3)

    elif poses.endswith(".pkl"):
        with open(poses, "rb") as f:
            poses = pickle.load(f)
            rots = poses["smpl_poses"] # (N, 72)
            rots = rots.reshape(-1, 24, 3) # (N, 24, 3)
            scaling = poses["smpl_scaling"]  # (1,)
            trans = poses["smpl_trans"]  # (N, 3)
    
    else:
        raise Exception("This file type is not supported!")

    if scaling is not None:
        trans /= scaling
    
    # to quaternion
    rots = quat.from_axis_angle(rots)
    
    pos = offsets[None].repeat(len(rots), axis=0)
    positions = pos.copy()
    positions[:,0] += trans * 100
    rotations = np.degrees(quat.to_euler(rots, order=order))
    
    if model_type == "smplx":
        offsets = np.concatenate([np.zeros((1, 3)), offsets])
        positions = np.concatenate([np.zeros((2064, 1, 3)), positions], axis=1)
        rotations = np.concatenate([np.zeros((2064, 1, 3)), rotations], axis=1)
        parents = np.insert(parents+1, 0, -1)

    bvh_data ={
        "rotations": rotations,
        "positions": positions,
        "offsets": offsets,
        "parents": parents,
        "names": names[model_type],
        "order": order,
        "frametime": 1 / fps,
    }
    
    if not output.endswith(".bvh"):
        output = output + ".bvh"
    
    bvh.save(output, bvh_data)
    
    if mirror:
        rots_mirror, trans_mirror = mirror_rot_trans(
                rots, trans, names[model_type], parents)
        positions_mirror = pos.copy()
        positions_mirror[:,0] += trans_mirror
        rotations_mirror = np.degrees(
            quat.to_euler(rots_mirror, order=order))
        
        bvh_data ={
            "rotations": rotations_mirror,
            "positions": positions_mirror,
            "offsets": offsets,
            "parents": parents,
            "names": names[model_type],
            "order": order,
            "frametime": 1 / fps,
        }
        
        output_mirror = output.split(".")[0] + "_mirror.bvh"
        bvh.save(output_mirror, bvh_data)

if __name__ == "__main__":
    args = parse_args()

    smpl2bvh(model_path=args.model_path, model_type=args.model_type, order=args.order,
             mirror=args.mirror, gender=args.gender,
             poses=args.poses, num_betas=args.num_betas, 
             fps=args.fps, output=args.output)
    
    print("finished!")
