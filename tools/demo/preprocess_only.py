import os
import shutil
import numpy as np
import cv2
import torch
import argparse
from hmr4d.utils.pylogger import Log
import hydra
from hydra import initialize_config_module, compose
from pathlib import Path
import subprocess
from torch.utils.data import DataLoader, ConcatDataset

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    read_video_np,
    save_video,
    get_writer,
    get_video_reader,
)
from hmr4d.utils.vis.cv2_utils import draw_coco17_skeleton_batch, draw_coco133_skeleton_batch, draw_bbx_xyxy_on_image_batch_multiperson

from hmr4d.utils.preproc import Tracker, VitPoseWholebodyExtractor, SLAMModel

from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy_batch, estimate_K, convert_K_to_K4
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from tqdm import tqdm

from hmr4d.utils.datasets.vitdet_dataset import ViTDetDataset, recursive_to

from hamer.models import load_hamer, DEFAULT_CHECKPOINT_HAMER


CRF = 23  # 17 is lossless, every +6 halves the mp4 size
def get_video_fps(video_path):
    reader = cv2.VideoCapture(video_path)
    fps = reader.get(cv2.CAP_PROP_FPS)
    reader.release()
    return fps


def parse_args_to_cfg():
    # Put all args to cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="inputs/demo/dance_3.mp4")
    parser.add_argument("--video_name", type=str, default=None, help="by default to video.stem")
    parser.add_argument("--output_root", type=str, default=None, help="by default to outputs/demo")
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip DPVO")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for VitPose and VitFeat")
    parser.add_argument("--recreate_video", action="store_true", help="If true, encode the original video to 30 fps for visualization")
    parser.add_argument("--verbose", action="store_true", help="If true, draw intermediate results")
    args = parser.parse_args()

    # Input
    video_path = Path(args.video)
    fps = get_video_fps(video_path) if not args.recreate_video else 30
    assert video_path.exists(), f"Video not found at {video_path}"
    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input]: {video_path}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")
    # Cfg
    with initialize_config_module(version_base="1.3", config_module=f"hmr4d.configs"):
        overrides = [
            f"video_name={video_path.stem}" if args.video_name is None else f"video_name={args.video_name}",
            f"static_cam={args.static_cam}",
            f"verbose={args.verbose}",
            f"+batch_size={args.batch_size}",
            f"+fps={round(fps)}",
        ]

        # Allow to change output root
        if args.output_root is not None:
            overrides.append(f"output_root={args.output_root}")
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    # Output
    Log.info(f"[Output Dir]: {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # Copy raw-input-video to video_path
    Log.info(f"[Copy Video] {video_path} -> {cfg.video_path}")
    if not Path(cfg.video_path).exists() or get_video_lwh(video_path)[0] != get_video_lwh(cfg.video_path)[0]:
        # create a soft link
        if args.recreate_video:
            reader = get_video_reader(video_path)
            writer = get_writer(cfg.video_path, fps=30, crf=CRF)
            for img in tqdm(reader, total=get_video_lwh(video_path)[0], desc=f"Copy"):
                writer.write_frame(img)
            writer.close()
            reader.close()
        else:
            try:
                os.symlink(os.path.abspath(video_path), os.path.abspath(cfg.video_path))
            except FileExistsError:
                os.remove(os.path.abspath(cfg.video_path))
                os.symlink(os.path.abspath(video_path), os.path.abspath(cfg.video_path))
    
    valid_video_path = os.path.abspath(cfg.video_path).replace('0_input_video.mp4', 'valid_video.mp4')
    try:
        shutil.copy2(os.path.abspath(cfg.video_path), valid_video_path)
    except FileExistsError:
        os.remove(valid_video_path)
        shutil.copy2(os.path.abspath(cfg.video_path), valid_video_path)
        
    return cfg


@torch.no_grad()
def run_preprocess(cfg):
    Log.info(f"[Preprocess] Start!")
    tic = Log.time()
    video_path = cfg.video_path
    paths = cfg.paths
    static_cam = cfg.static_cam
    verbose = cfg.verbose
    batch_size = cfg.batch_size

    # Get bbx tracking result
    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy, bbx_conf = tracker.get_all_tracks(video_path, frame_thres=0.5)  # (P, L, 4), (P, L) discard short tracks
        bbx_xys = get_bbx_xys_from_xyxy_batch(bbx_xyxy, base_enlarge=1.2).float()  # (P, L, 3) apply aspect ratio and enlarge
        torch.save({"bbx_xyxy": bbx_xyxy.detach().cpu(), "bbx_xys": bbx_xys.detach().cpu(), "bbx_conf": bbx_conf.detach().cpu()}, paths.bbx)
        del tracker
    else:
        bbx_xys = torch.load(paths.bbx, weights_only=True)["bbx_xys"]
        Log.info(f"[Preprocess] bbx (xyxy, xys) from {paths.bbx}")
    if verbose:
        video = read_video_np(video_path)
        bbx_xyxy = torch.load(paths.bbx, weights_only=True)["bbx_xyxy"]
        video_overlay = draw_bbx_xyxy_on_image_batch_multiperson(bbx_xyxy, video)
        save_video(video_overlay, cfg.paths.bbx_xyxy_video_overlay, fps=cfg.fps)
    person_num = bbx_xys.shape[0]
    print(f"person_num: {person_num}")
    def chunk_first_axis(tensor, person_num):
        # (frame_num*person_num, ..) -> (frame_num, person_num, ..)
        return tensor.reshape(-1, person_num, *tensor.shape[1:])
    
    # Get VitPose-wholebody (0-16: 17 body keypoints, 17-22: 6 foot keypoints, 23-90: 68 face keypoints, 91-132: 42 hand keypoints)
    if not Path(paths.vitpose_wholebody).exists():
        vitpose_extractor = VitPoseWholebodyExtractor(batch_size=batch_size)
        vitpose_wholebody, cropped_imgs = vitpose_extractor.extract_multiperson(video_path, bbx_xys)  # (P, F, 133, 3)
        torch.save(vitpose_wholebody.detach().cpu(), paths.vitpose_wholebody)
        del vitpose_extractor
    else:
        vitpose_wholebody = torch.load(paths.vitpose_wholebody, weights_only=True)
        Log.info(f"[Preprocess] vitpose-wholebody from {paths.vitpose_wholebody}")
    if verbose:
        video = read_video_np(video_path)
        video_overlay = draw_coco133_skeleton_batch(video, vitpose_wholebody.transpose(0, 1), 0.5, 2, 4)
        save_video(video_overlay, paths.vitpose_wholebody_video_overlay, fps=cfg.fps)
    
    # Get VitPose
    if not Path(paths.vitpose).exists():
        vitpose = vitpose_wholebody[:, :, :17, :]  # (P, F, 17, 3)
        torch.save(vitpose.detach().cpu(), paths.vitpose)
    else:
        vitpose = torch.load(paths.vitpose, weights_only=True)
        Log.info(f"[Preprocess] vitpose from {paths.vitpose}")
    if verbose:
        video = read_video_np(video_path)
        video_overlay = draw_coco17_skeleton_batch(video, vitpose.transpose(0, 1), 0.5)
        save_video(video_overlay, paths.vitpose_video_overlay, fps=cfg.fps)
    
    # Get mano params
    if not Path(paths.mano_params).exists():
        hamer_model, model_cfg_hamer = load_hamer(DEFAULT_CHECKPOINT_HAMER)  # setup HaMeR model
        hamer_model = hamer_model.cuda()
        hamer_model.eval()
        # create hamer dataset
        frames = read_frames(video_path)
        hamer_dataloader = load_images(frames, vitpose_wholebody, model_cfg_hamer)
        all_mano_params = {'left_hand_global_orient': [], 'left_hand_pose': [], 'left_hand_valid': [], 
                        'right_hand_global_orient': [], 'right_hand_pose': [], 'right_hand_valid': []}
        for batch in tqdm(hamer_dataloader, desc="Hamer"):
            batch = recursive_to(batch, target="cuda")
            mano_poses = predict_mano(batch, hamer_model)
            for k, v in mano_poses.items():
                all_mano_params[k].append(chunk_first_axis(v, person_num))
        for k, v in all_mano_params.items():
            all_mano_params[k] = torch.cat(v, dim=0).transpose(0, 1).detach().cpu()
        torch.save(all_mano_params, paths.mano_params)
    else:
        Log.info(f"[Preprocess] mano_params from {paths.mano_params}")

    # Get DPVO results
    if not static_cam:  # use slam to get cam rotation
        if not Path(paths.slam).exists():
            length, width, height = get_video_lwh(cfg.video_path)
            K_fullimg = estimate_K(width, height)
            intrinsics = convert_K_to_K4(K_fullimg)
            slam = SLAMModel(video_path, width, height, intrinsics, buffer=4000, resize=0.5)
            bar = tqdm(total=length, desc="DPVO")
            while True:
                ret = slam.track()
                if ret:
                    bar.update()
                else:
                    break
            slam_results = slam.process()  # (L, 7), numpy
            torch.save(slam_results, paths.slam)
        else:
            Log.info(f"[Preprocess] slam results from {paths.slam}")

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time()-tic:.2f}s")


def fetch_smpl_params(all_person_dict, person_idx):
    return {k: v[person_idx] for k, v in all_person_dict.items()}


def read_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def load_images(cv2_images, wholebody_kpts, model_cfg):
    hand_datasets = []

    for idx, img_cv2 in enumerate(cv2_images):
        vitposes_out = wholebody_kpts[:, idx]  # (P, F, 133, 3) -> (P, 133, 3)
        person_num = vitposes_out.shape[0]
        bboxes, is_right, is_valid = [], [], []
        for person_idx in range(person_num):
            vitposes = vitposes_out[person_idx]
            left_hand_keyp = vitposes[-42:-21]
            right_hand_keyp = vitposes[-21:]

            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
            else:
                bbox = [keyp[:, 0].min(), keyp[:, 1].min(), keyp[:, 0].max(), keyp[:, 1].max()]
            bboxes.append(bbox)
            is_right.append(0)
            is_valid.append(sum(valid) > 3)
            
            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
            else:
                bbox = [keyp[:, 0].min(), keyp[:, 1].min(), keyp[:, 0].max(), keyp[:, 1].max()]
            bboxes.append(bbox)
            is_right.append(1)
            is_valid.append(sum(valid) > 3)

        if len(bboxes) == 0:
            bboxes, right, valid = np.empty((0, 4)), np.empty(0), np.empty(0)
        else:
            bboxes, right, valid = np.array(bboxes), np.array(is_right), np.array(is_valid)
            
        hand_dataset = ViTDetDataset(model_cfg, img_cv2, bboxes, right, valid, rescale_factor=2.0)
        hand_datasets.append(hand_dataset)

    concatenated_hand_dataset = ConcatDataset(hand_datasets)
    hand_dataloader = DataLoader(concatenated_hand_dataset, batch_size=64, shuffle=False, num_workers=0)
    return hand_dataloader


def predict_mano(batch, model):
    batch_size = batch['img'].shape[0]
    with torch.no_grad():
        out = model(batch)
    flip_idx = batch['right'] == 0
    invalid_idx = batch['valid'] == 0   # might be invisible, set as default hand pose
    assert flip_idx.sum() == batch_size // 2, f"flip_idx: {flip_idx.sum()}, batch_size: {batch_size}"
    
    mano_params = out['pred_mano_params']
    global_orient = mano_params['global_orient'].reshape(batch_size, -1, 3, 3)
    hand_pose = mano_params['hand_pose'].reshape(batch_size, -1, 3, 3)
    hand_pose[invalid_idx] = torch.zeros_like(hand_pose[invalid_idx])
    # Flip the hand pose
    reflection_matrix = torch.tensor([
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, 1]
    ], dtype=torch.float32).cuda()
    global_orient[flip_idx] = torch.einsum('bkij,ij->bkij', global_orient[flip_idx], reflection_matrix)
    hand_pose[flip_idx] = torch.einsum('bkij,ij->bkij', hand_pose[flip_idx], reflection_matrix)
    mano_poses = {
        'left_hand_global_orient': global_orient[flip_idx],
        'left_hand_pose': hand_pose[flip_idx],
        'right_hand_global_orient': global_orient[~flip_idx],
        'right_hand_pose': hand_pose[~flip_idx],
        'left_hand_valid': batch['valid'][flip_idx],
        'right_hand_valid': batch['valid'][~flip_idx]
    }
    return mano_poses


if __name__ == "__main__":
    cfg = parse_args_to_cfg()
    paths = cfg.paths
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f'[GPU]: {torch.cuda.get_device_properties("cuda")}')

    # ===== Preprocess and save to disk ===== #
    run_preprocess(cfg)
    # delete 0_input_video.mp4
    subprocess.run(["rm", cfg.video_path])
    
"""
CUDA_VISIBLE_DEVICES=2, python -m tools.demo.preprocess_only --video=docs/example_video/vertical_dance.mp4 --output_root outputs/demo_mp -s
CUDA_VISIBLE_DEVICES=7, python -m tools.demo.preprocess_only --video=docs/example_video/two_persons.mp4 --output_root outputs/demo_mp_hands --verbose
CUDA_VISIBLE_DEVICES=7, python -m tools.demo.preprocess_only --video=/mnt/data/jing/Video_Generation/video_data_repos/video_smplx_labeling/sapiens/example_data2/video00024_010.mp4 --output_root outputs/demo --verbose
CUDA_VISIBLE_DEVICES=3, python -m tools.demo.preprocess_only --video=/mnt/data/jing/Video_Generation/video_data_repos/video_preprocessor/WHAM/examples/dance2.mp4 --output_root outputs/demo_mp
"""