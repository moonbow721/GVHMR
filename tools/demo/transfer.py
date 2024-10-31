import os
import numpy as np
import torch
import argparse
from pathlib import Path



def retarget_transl(transl_motion, transl_origin):
    """
    Retarget the start point of the transl_motion to the start point of the transl_origin.
    
    Args:
        transl_motion (torch.Tensor): The translation motion, shape (P, F, 3).
        transl_origin (torch.Tensor): The translation origin, shape (P, 3).
    Returns:
        torch.Tensor: The retargeted translation motion, shape (P, F, 3).
    """
    retargeted_transl = transl_motion.clone()
    retargeted_transl[:, 0, :] = transl_origin[:, :]
    retargeted_transl[:, 1:, :] = transl_motion[:, 1:, :] + (transl_origin[:, None, :] - transl_motion[:, 0:1, :])
    return retargeted_transl


def transfer_parameters(source_data, reference_data, args):
    """Transfer parameters from reference to source motion sequence"""
    person_num_source = source_data['smpl_params_incam']['transl'].shape[0]
    frame_num = source_data['smpl_params_incam']['transl'].shape[1]
    person_num_ref = reference_data['smpl_params_incam']['transl'].shape[0]
    assert person_num_source == person_num_ref, f"Source and reference must have the same number of persons, but got {person_num_source} and {person_num_ref}"
    
    # Get required parameters from reference
    ref_smpl_dict = reference_data['smpl_params_incam']
    ref_betas = ref_smpl_dict['betas'][:, 0]   # (P, 10)
    ref_transl_origin = ref_smpl_dict['transl'][:, 0]  # (P, 3)
    
    ref_K_fullimg = reference_data['K_fullimg'][0]
    ref_width = reference_data['width'][0]
    ref_height = reference_data['height'][0]
    ref_focal_length = reference_data['focal_length'][0]
    
    # Create output dict
    output_data = {'fps': source_data['fps'],
                   'smpl_params_incam': source_data['smpl_params_incam'].copy()}
    
    # Transfer translation motion
    if args.transl_type == "retarget":
        output_data['smpl_params_incam']['transl'] = retarget_transl(source_data['smpl_params_incam']['transl'], ref_transl_origin)
    elif args.transl_type == "static":
        output_data['smpl_params_incam']['transl'] = ref_transl_origin[:, None, :].repeat(1, frame_num, 1)  
    
    # Transfer shape parameters if requested
    if args.figure_transfer:
        output_data['smpl_params_incam']['betas'] = ref_betas[:, None, :].repeat(1, frame_num, 1) 
        
    # Transfer camera parameters if requested  
    if args.view_transfer:
        output_data['K_fullimg'] = ref_K_fullimg[None, :, :].repeat(frame_num, 1, 1)  
        output_data['width'] = ref_width[None].repeat(frame_num)
        output_data['height'] = ref_height[None].repeat(frame_num)
        output_data['focal_length'] = ref_focal_length[None].repeat(frame_num)
    else:
        output_data['K_fullimg'] = source_data['K_fullimg']
        output_data['width'] = source_data['width']
        output_data['height'] = source_data['height']
        output_data['focal_length'] = source_data['focal_length']
        
    return output_data


def main(args):
    device = torch.device(f"cuda:{args.device}")
    
    # Load reference data
    reference_data = torch.load(args.reference_path, map_location=device)
    
    # Load source motion
    source_data = torch.load(args.driving_motion_path, map_location=device)
    
    # Transfer parameters
    output_data = transfer_parameters(source_data, reference_data, args)
    
    # Save transferred results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_data, output_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="transfer smpl")
    parser.add_argument("--body_model_dir", type=str, default="inputs/checkpoints/body_models/", help="Path to smplx model dir")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--driving_motion_path", type=str,
        default="driving_videos/source_motion_001/hmr4d_results.pt",
        help="Path to smpl results of source motion",
    )
    parser.add_argument(
        "--reference_path", type=str,
        default="reference_imgs/ref_img001/hmr4d_results.pt",
        help="Path to smpl results of reference img (duplicated motion, just repeat the same frame)",
    )
    parser.add_argument("--output_path", type=str, default="transferred_motion/source_motion_001_ref_img001.pt", help="Path to output motion sequence")
    parser.add_argument("--transl_type", choices=["static", "retarget"], default="retarget", help="Type of merged translation")
    parser.add_argument("--figure_transfer", action="store_true", help="Transfer SMPL shape parameters")
    parser.add_argument("--view_transfer", action="store_true", help="Transfer camera parameters")
    
    args = parser.parse_args()
    main(args)