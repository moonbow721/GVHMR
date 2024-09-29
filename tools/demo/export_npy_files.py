import os
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import smplx  # Ensure smplx is installed: pip install smplx
from hmr4d.utils.smplx_utils import make_smplx


def convert_smpl_to_vertices(smpl_model, global_orient, body_pose, betas, device):
    """
    Convert SMPL parameters to vertices using the SMPLX model on CUDA.

    Args:
        smpl_model (smplx.SMPL): Initialized SMPLX model on CUDA.
        transl (torch.Tensor): Translation tensor, shape (3,).
        global_orient (torch.Tensor): Global orientation in axis-angle, shape (3,).
        body_pose (torch.Tensor): Body pose in axis-angle, shape (63,).
        betas (torch.Tensor): Shape coefficients, shape (10,).

    Returns:
        torch.Tensor: Vertices of the SMPL model, shape (n_verts, 3), on CUDA.
    """
    # Move tensors to CUDA
    global_orient = global_orient.to(device).float()
    body_pose = body_pose.to(device).float()
    betas = betas.to(device).float()

    # Forward pass to get vertices
    with torch.no_grad():
        output = smpl_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas
        )
    vertices = output.vertices  # (num_persons, n_verts, 3)

    return vertices


def convert_hmr4d_to_npy(input_file, output_dir, device):
    """
    Convert hmr4d_results.pt to a batch of .npy files.

    Args:
        input_file (str): Path to hmr4d_results.pt file.
        output_dir (str): Directory where .npy files will be saved.
        device (torch.device): CUDA device for processing.
    """
    # Load the hmr4d_results.pt file
    data = torch.load(input_file, map_location=device)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract relevant data
    smpl_params = data['smpl_params_incam']
    num_frames = smpl_params['transl'].shape[1]
    num_persons = smpl_params['transl'].shape[0]

    print(f"Number of frames: {num_frames}, Number of persons: {num_persons}")

    # Initialize SMPLX model
    smpl_model = make_smplx("supermotion").to(device) # n_verts = 10475

    # Process each frame
    for frame in tqdm(range(num_frames), desc="Processing frames"):
        frame_data = {
            'verts': [],
            'cam_t': []
        }

        # Batch processing for all persons in the current frame
        transl = smpl_params['transl'][:, frame]           # (num_persons, 3)
        global_orient = smpl_params['global_orient'][:, frame]  # (num_persons, 3)
        body_pose = smpl_params['body_pose'][:, frame]     # (num_persons, 63)
        betas = smpl_params['betas'][:, frame]             # (num_persons, 10)

        # Convert all parameters to vertices in a single batch
        vertices = convert_smpl_to_vertices(
            smpl_model,
            global_orient,
            body_pose,
            betas,
            device
        )  # (num_persons, n_verts, 3)

        # Move vertices to CPU and convert to numpy
        vertices_np = vertices.cpu().numpy()

        # Append to frame_data
        for person_idx in range(num_persons):
            frame_data['verts'].append(vertices_np[person_idx])
            frame_data['cam_t'].append(transl[person_idx].cpu().numpy())

        # Save the frame data as a .npy file
        output_file = os.path.join(output_dir, f'{frame:04d}.npy')
        np.save(output_file, frame_data)

    print(f"Conversion complete. .npy files saved in {output_dir}")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    convert_hmr4d_to_npy(args.input, args.output, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert hmr4d_results.pt to .npy files")
    parser.add_argument("--input", required=True, help="Path to hmr4d_results.pt file")
    parser.add_argument("--output", required=True, help="Output directory for .npy files")
    parser.add_argument("--device", default="cuda", help="Device to use for rendering (e.g., 'cuda:0' or 'cpu')")
    args = parser.parse_args()

    main(args)
    print(f"All .npy files have been successfully exported to {args.output}")

'''
python -m tools.demo.export_npy_files --input /mnt/data/jing/Video_Generation/video_data_repos/video_preprocessor/GVHMR/outputs/demo_mp/vertical_dance/hmr4d_results.pt --output /mnt/data/jing/Video_Generation/video_data_repos/video_preprocessor/GVHMR/outputs/demo_mp/vertical_dance/smpl_results
'''