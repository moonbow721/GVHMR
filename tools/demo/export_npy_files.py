import os
import numpy as np
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from smplx import SMPLXLayer

def merge_mano_to_smpl(smpls, manos, replace_wrist=True):
    # smplx body pose: [batchsize, 21, 3, 3] (rotation matrix)
    batch_size = smpls['body_pose'].shape[0]
    
    smplx_params = {k: v.clone() for k, v in smpls.items()}
    global_orient = smplx_params['global_orient']
    smplx_params['left_hand_pose'] = manos['left_hand_pose']
    smplx_params['right_hand_pose'] = manos['right_hand_pose']
    if replace_wrist:
        left_wrist_chain = np.array([3, 6, 9, 13, 16, 18]) - 1  # pelvis (-1, global orient) -> spine1 -> spine2 -> spine3 -> left_collar -> left_shoulder -> left_elbow
        right_wrist_chain = np.array([3, 6, 9, 14, 17, 19]) - 1  # pelvis (-1, global orient) -> spine1 -> spine2 -> spine3 -> right_collar -> right_shoulder -> right_elbow
        left_wrist_pose = global_orient.clone()
        for idx in left_wrist_chain:
            left_wrist_pose = torch.matmul(left_wrist_pose, smplx_params['body_pose'][:, idx:idx+1])
        right_wrist_pose = global_orient.clone()
        for idx in right_wrist_chain:
            right_wrist_pose = torch.matmul(right_wrist_pose, smplx_params['body_pose'][:, idx:idx+1])
        
        left_idx, right_idx = manos['left_hand_valid'] == 1, manos['right_hand_valid'] == 1
        smplx_params['body_pose'][left_idx, 19:20] = torch.matmul(torch.inverse(left_wrist_pose[left_idx]), manos['left_hand_global_orient'][left_idx])
        smplx_params['body_pose'][right_idx, 20:21] = torch.matmul(torch.inverse(right_wrist_pose[right_idx]), manos['right_hand_global_orient'][right_idx])
    
    # set transl to zero for canonicalization in `verts`
    smplx_params['transl'] = torch.zeros_like(smplx_params['transl'])
    
    return smplx_params


def fetch_frame_dict(all_dict, frame_idx):
    frame_dict = {}
    for k, v in all_dict.items():
        frame_dict[k] = v[:, frame_idx]
    return frame_dict
    

def convert_hmr4d_to_npy(input_file, output_dir, smplx_model_path, mano_params_file, device):
    """
    Convert hmr4d_results.pt to a batch of .npy files.

    Args:
        input_file (str): Path to hmr4d_results.pt file.
        output_dir (str): Directory where .npy files will be saved.
        smplx_model_path (str): Path to SMPLX model dir.
        mano_params_file (str): Path to mano_params.pt file.
        device (torch.device): CUDA device for processing.
    """
    # Load the hmr4d_results.pt file
    data = torch.load(input_file, map_location=device)

    # Load mano params if provided
    mano_params = None
    if mano_params_file:
        mano_params = torch.load(mano_params_file, map_location=device)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Build smplx model
    smplx_model = SMPLXLayer(model_path=smplx_model_path, use_pca=False).to(device)
    
    # Extract relevant data
    smpl_params = data['smpl_params_incam']
    num_frames = smpl_params['transl'].shape[1]
    num_persons = smpl_params['transl'].shape[0]

    print(f"Number of frames: {num_frames}, Number of persons: {num_persons}")

    # Process each frame
    for frame in tqdm(range(num_frames), desc="Processing frames"):
        frame_data = {
            'verts': [],
            'cam_t': []
        }

        # Batch processing for all persons in the current frame
        transl = smpl_params['transl'][:, frame]           # (num_persons, 3)
        # global_orient = smpl_params['global_orient'][:, frame]  # (num_persons, 1, 3, 3)
        # body_pose = smpl_params['body_pose'][:, frame]     # (num_persons, 1, 21, 3)
        # betas = smpl_params['betas'][:, frame]             # (num_persons, 10)

        # Merge mano params if available
        if mano_params:
            smplx_params = merge_mano_to_smpl(fetch_frame_dict(smpl_params, frame), 
                                              fetch_frame_dict(mano_params, frame))
        else:
            smplx_params = fetch_frame_dict(smpl_params, frame)

        # Convert all parameters to vertices in a single batch
        vertices = smplx_model(**smplx_params).vertices.detach().cpu().numpy()

        # Append to frame_data
        for person_idx in range(num_persons):
            frame_data['verts'].append(vertices[person_idx])  # canonicalized
            frame_data['cam_t'].append(transl[person_idx].cpu().numpy())  # camera translation

        # Save the frame data as a .npy file
        output_file = os.path.join(output_dir, f'{frame:04d}.npy')
        np.save(output_file, frame_data)

    print(f"Conversion complete. .npy files saved in {output_dir}")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    convert_hmr4d_to_npy(args.input, args.output, args.smplx_model_path, args.mano_params, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert hmr4d_results.pt to .npy files")
    parser.add_argument("--input", required=True, help="Path to hmr4d_results.pt file")
    parser.add_argument("--output", required=True, help="Output directory for .npy files")
    parser.add_argument("--mano_params", type=str, default=None, help="Path to mano_params.pt file")
    parser.add_argument("--smplx_model_path", type=str, default="inputs/checkpoints/body_models/smplx/", help="Path to smplx model dir")
    parser.add_argument("--device", default="cuda", help="Device to use for rendering (e.g., 'cuda:0' or 'cpu')")
    args = parser.parse_args()

    main(args)
    print(f"All .npy files have been successfully exported to {args.output}")

'''
CUDA_VISIBLE_DEVICES=7 python -m tools.demo.export_npy_files --input ./outputs/demo_mp_hands/two_persons/hmr4d_results.pt --output ./outputs/demo_mp_hands/two_persons/smpl_results --mano_params ./outputs/demo_mp_hands/two_persons/mano_params.pt
'''