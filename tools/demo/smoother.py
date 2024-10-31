import numpy as np
import torch
import argparse
import os
import subprocess
from pathlib import Path

def gaussian_weights(window_size, sigma=1.0):
    """
    Generate Gaussian weights for a given window size and standard deviation.
    
    :param window_size: The size of the sliding window.
    :param sigma: The standard deviation of the Gaussian distribution.
    :return: A list of Gaussian weights.
    """
    radius = window_size // 2
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    weights = torch.exp(-x**2 / (2 * sigma**2))
    return weights / weights.sum()

def weighted_average_smooth(data, window_size=5, weights=None):
    """
    Smooth the data using a weighted average in a sliding window.
    
    :param data: The input data to smooth (e.g., body_pose, global_orient), the first dimension is the sequence length.
    :param window_size: The size of the sliding window.
    :param weights: A list of weights for the sliding window.
    :return: The smoothed data, same shape as the input data.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")

    if weights is None:
        weights = gaussian_weights(window_size).to(data.device)
    else:
        weights = torch.tensor(weights, dtype=torch.float32).to(data.device)
        weights = weights / weights.sum()  # Normalize weights to sum to 1

    half_window = window_size // 2
    smoothed_data = torch.zeros_like(data)

    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        
        current_weights = weights[half_window - (i - start):half_window + (end - i)]
        # If the window is at the beginning or end of the sequence, adjust the weights
        current_weights = current_weights / current_weights.sum()

        # Expand dimensions of current_weights to match the data dimensions
        expanded_weights = current_weights.view(-1, *([1] * (data.dim() - 1)))

        # Weighted sum for the window
        smoothed_data[i] = (data[start:end] * expanded_weights).sum(dim=0)

    return smoothed_data

def smooth_parameters(data_dict, dict_type='hmr', window_size=5, weights=None):
    """
    Smooth SMPL parameters in place using Gaussian smoothing.
    
    Parameters to be smoothed:
    For HMR results:
    - smpl_params_incam:
        - body_pose: (P, L, 23, 3, 3) - Body joint rotations
        - global_orient: (P, L, 1, 3, 3) - Global body orientation
        - transl: (P, L, 3) - Global body translation
        - betas: (P, L, 10) - Body shape parameters
    
    For MANO results:
    - right_hand_pose: (P, L, 15, 3, 3) - Right hand joint rotations
    - left_hand_pose: (P, L, 15, 3, 3) - Left hand joint rotations
    - right_hand_global_orient: (P, L, 1, 3, 3) - Right hand global orientation
    - left_hand_global_orient: (P, L, 1, 3, 3) - Left hand global orientation
    
    :param data_dict: Dictionary containing SMPL parameters to smooth
    :param window_size: Size of smoothing window
    :param weights: Optional custom weights for smoothing window
    """
    assert dict_type in ['hmr', 'mano'], "dict_type must be either 'hmr' or 'mano'"
    # Parameters to smooth for HMR results
    hmr_params = ['body_pose', 'global_orient', 'transl', 'betas']
    
    # Parameters to smooth for MANO results
    mano_params = ['right_hand_pose', 'left_hand_pose', 
                  'right_hand_global_orient', 'left_hand_global_orient']

    # Check if this is HMR or MANO data
    if dict_type == 'hmr':
        # Smooth HMR parameters
        for key in hmr_params:
            if key in data_dict['smpl_params_incam']:
                param = data_dict['smpl_params_incam'][key].transpose(0, 1)
                smoothed = weighted_average_smooth(param, window_size, weights)
                data_dict['smpl_params_incam'][key] = smoothed.transpose(0, 1)
    elif dict_type == 'mano':
        # Smooth MANO parameters
        for key in mano_params:
            if key in data_dict:
                param = data_dict[key].transpose(0, 1)
                smoothed = weighted_average_smooth(param, window_size, weights)
                data_dict[key] = smoothed.transpose(0, 1)

    return data_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Smooth SMPL parameters')
    parser.add_argument('--window_size', type=int, default=5, help='Size of smoothing window (must be odd)')
    parser.add_argument('--sigma', type=float, default=1.0, help='Sigma for Gaussian weights')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing SMPL parameters')
    parser.add_argument('--export_verts', action='store_true', help='Export smoothed SMPL vertices')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run smoothing on')
    args = parser.parse_args()

    # Process each file in data directory
    data_dir = Path(args.data_dir)
    
    # Process HMR results
    hmr_files = ['hmr4d_results.pt', 'hmr2_results.pt']
    weights=gaussian_weights(args.window_size, args.sigma)
    for hmr_file in hmr_files:
        hmr_path = data_dir / hmr_file
        if hmr_path.exists():
            print(f"Processing {hmr_file}...")
            data = torch.load(hmr_path, map_location=args.device)
            smooth_parameters(data, dict_type='hmr', window_size=args.window_size, weights=weights)
            save_path = data_dir / hmr_file.replace('.pt', '_smoothed.pt')
            torch.save(data, save_path)
            print(f"Saved smoothed parameters to {save_path}")
            
    # Process MANO parameters if they exist
    mano_path = data_dir / 'mano_params.pt'
    weights=gaussian_weights(args.window_size, args.sigma)
    if mano_path.exists():
        print("Processing mano_params.pt...")
        data = torch.load(mano_path, map_location=args.device)
        smooth_parameters(data, dict_type='mano', window_size=args.window_size, weights=weights)
        save_path = data_dir / 'mano_params_smoothed.pt'
        torch.save(data, save_path)
        print(f"Saved smoothed parameters to {save_path}")
    
    # Process SMPL vertices if HMR results exist
    if args.export_verts:
        verts_pt_path = os.path.join(data_dir, "smpl_verts_smoothed.pt")
        for hmr_file in hmr_files:
            hmr_smoothed_path = data_dir / hmr_file.replace('.pt', '_smoothed.pt')
            mano_smoothed_path = data_dir / 'mano_params_smoothed.pt'
            if hmr_smoothed_path.exists():
                print(f"Generating smoothed SMPL vertices from {hmr_smoothed_path}...")
                mano_path = str(mano_smoothed_path) if mano_smoothed_path.exists() else None
                
                cmd = ["python", "-m", "tools.demo.export_pt_verts", 
                    "--input", str(hmr_smoothed_path),
                    "--output", verts_pt_path,
                    "--smpl_type", "smplx" if "hmr4d" in hmr_file else "smplh",   # hmr4d_results.pt uses smplx, hmr2_results.pt uses smplh
                    "--device", args.device]
                
                if mano_path:
                    cmd.extend(["--mano_params", mano_path])
                    
                subprocess.run(cmd)
                print(f"Saved smoothed SMPL vertices to {verts_pt_path}")
                break  # Only need to process one HMR file, hmr4d_results.pt first

'''
python -m tools.demo.smoother --data_dir /mnt/data/jing/Video_Generation/Data/video_dataset_champ_debug/debug_gvhmr_folder/two_persons --window_size 5 --sigma 1.0 --device cuda:7
'''
