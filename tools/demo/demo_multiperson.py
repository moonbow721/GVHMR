import os
import cv2
import torch
import pytorch_lightning as pl
import numpy as np
import argparse
from hmr4d.utils.pylogger import Log
import hydra
from hydra import initialize_config_module, compose
from pathlib import Path
from pytorch3d.transforms import quaternion_to_matrix

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    read_video_np,
    save_video,
    merge_videos_horizontal,
    get_writer,
    get_video_reader,
)
from hmr4d.utils.vis.cv2_utils import draw_coco17_skeleton_batch, draw_bbx_xyxy_on_image_batch_multiperson

from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SLAMModel

from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy_batch, estimate_K, convert_K_to_K4, create_camera_sensor
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from tqdm import tqdm
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from einops import einsum, rearrange


CRF = 23  # 17 is lossless, every +6 halves the mp4 size


def parse_args_to_cfg():
    # Put all args to cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="inputs/demo/dance_3.mp4")
    parser.add_argument("--output_root", type=str, default=None, help="by default to outputs/demo")
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip DPVO")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for VitPose and VitFeat")
    parser.add_argument("--recreate_video", action="store_true", help="If true, encode the original video to 30 fps")
    parser.add_argument("--verbose", action="store_true", help="If true, draw intermediate results")
    args = parser.parse_args()

    # Input
    video_path = Path(args.video)
    assert video_path.exists(), f"Video not found at {video_path}"
    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input]: {video_path}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")
    # Cfg
    with initialize_config_module(version_base="1.3", config_module=f"hmr4d.configs"):
        overrides = [
            f"video_name={video_path.stem}",
            f"static_cam={args.static_cam}",
            f"verbose={args.verbose}",
            f"+batch_size={args.batch_size}",  # Add batch_size here
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
            os.system(f"ln -s {video_path} {cfg.video_path}")

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
        bbx_xyxy = tracker.get_all_tracks(video_path, frame_thres=0.5).float()  # (P, L, 4), discard short tracks
        bbx_xys = get_bbx_xys_from_xyxy_batch(bbx_xyxy, base_enlarge=1.2).float()  # (P, L, 3) apply aspect ratio and enlarge
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    else:
        bbx_xys = torch.load(paths.bbx)["bbx_xys"]
        Log.info(f"[Preprocess] bbx (xyxy, xys) from {paths.bbx}")
    print(f"person_num: {bbx_xys.shape[0]}")
    
    if verbose:
        video = read_video_np(video_path)
        bbx_xyxy = torch.load(paths.bbx)["bbx_xyxy"]
        video_overlay = draw_bbx_xyxy_on_image_batch_multiperson(bbx_xyxy, video)
        save_video(video_overlay, cfg.paths.bbx_xyxy_video_overlay)
        
    # Get VitPose
    if not Path(paths.vitpose).exists():
        vitpose_extractor = VitPoseExtractor(batch_size=batch_size)
        vitpose, cropped_imgs = vitpose_extractor.extract_multiperson(video_path, bbx_xys)  # (P, F, 17, 3)
        torch.save(vitpose, paths.vitpose)
        del vitpose_extractor
    else:
        vitpose = torch.load(paths.vitpose)
        Log.info(f"[Preprocess] vitpose from {paths.vitpose}")
    if verbose:
        video = read_video_np(video_path)
        video_overlay = draw_coco17_skeleton_batch(video, vitpose[0], 0.5)
        save_video(video_overlay, paths.vitpose_video_overlay)
        
    # Get vit features
    if not Path(paths.vit_features).exists():
        extractor = Extractor(batch_size=batch_size)
        vit_features = extractor.extract_video_features_multiperson(cropped_imgs, bbx_xys)  # (P, F, 1024)
        torch.save(vit_features, paths.vit_features)
        del extractor
    else:
        Log.info(f"[Preprocess] vit_features from {paths.vit_features}")

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


def load_data_dict(cfg):
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        traj = torch.load(cfg.paths.slam)
        traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
        R_w2c = quaternion_to_matrix(traj_quat).mT
    K_fullimg = estimate_K(width, height).repeat(length, 1, 1)
    # K_fullimg = create_camera_sensor(width, height, 26)[2].repeat(length, 1, 1)

    data = {
        "length": torch.tensor(length),
        "bbx_xys": torch.load(paths.bbx)["bbx_xys"],
        "kp2d": torch.load(paths.vitpose),
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": torch.load(paths.vit_features),
    }
    return data


def fetch_smpl_params(all_person_dict, person_idx):
    return {k: v[person_idx] for k, v in all_person_dict.items()}


def create_merged_faces(faces_smpl, person_num, vert_offset):
    """
    Create merged faces for rendering multiple persons.
    
    Args:
        faces_smpl (numpy.ndarray): The original faces of the SMPL model, shape (face_num, 3).
        person_num (int): The number of persons to be rendered.
        vert_offset (int): The vertex offset for the current person.
    Returns:
        numpy.ndarray: The merged faces, shape (face_num * person_num, 3).
    """
    merged_faces = []
    for i in range(person_num):
        merged_faces.append(faces_smpl + i * vert_offset)
    return np.concatenate(merged_faces, axis=0)
    

def retarget_transl(global_transl, incam_transl_start, xz_only=False):
    """
    Retarget the start point of the global translation to the start point of the incam translation.
    
    Args:
        global_transl (torch.Tensor): The global translation, shape (P, F, 3).
        incam_transl_start (torch.Tensor): The incam translation start point, shape (P, 3).
    Returns:
        torch.Tensor: The retargeted global translation, shape (P, F, 3).
    """
    if xz_only:
        index = [0, 2]
    else:
        index = [0, 1, 2]
    retargeted_transl = global_transl.clone()
    retargeted_transl[:, 0, index] = incam_transl_start[:, index]
    retargeted_transl[:, 1:, index] = global_transl[:, 1:, index] + (incam_transl_start[:, None, index] - global_transl[:, 0:1, index])
    return retargeted_transl


def render_incam(cfg):
    incam_video_path = Path(cfg.paths.incam_video)
    if incam_video_path.exists():
        Log.info(f"[Render Incam] Video already exists at {incam_video_path}")
        return

    pred = torch.load(cfg.paths.hmr4d_results)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()   # (6890, 10475) 
    faces_smpl = make_smplx("smpl").faces   # (face_num, 3)

    # smpl
    # import ipdb; ipdb.set_trace()
    person_num = pred["smpl_params_incam"]["transl"].shape[0]
    frame_num = pred["smpl_params_incam"]["transl"].shape[1]
    merged_verts = []
    for person_idx in range(person_num):
        smplx_out = smplx(**to_cuda(fetch_smpl_params(pred["smpl_params_incam"], person_idx)))
        pred_c_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])  # (F, 6890, 3)
        merged_verts.append(pred_c_verts)
    pred_c_verts = torch.stack(merged_verts, dim=1).reshape(frame_num, -1, 3)  # (F, P, 6890, 3) -> (F, P*6890, 3)
    faces_smpl = create_merged_faces(faces_smpl, person_num, smplx2smpl.shape[0])
    
    # -- rendering code -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    K = pred["K_fullimg"][0]

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    reader = get_video_reader(video_path)  # (F, H, W, 3), uint8, numpy
    bbx_xys_render = torch.load(cfg.paths.bbx)["bbx_xys"]

    # -- render mesh -- #
    verts_incam = pred_c_verts  # (F, V, 3)
    writer = get_writer(incam_video_path, fps=30, crf=CRF)
    for i, img_raw in tqdm(enumerate(reader), total=get_video_lwh(video_path)[0], desc=f"Rendering Incam"):
        img = renderer.render_mesh(verts_incam[i].cuda(), img_raw, [0.8, 0.8, 0.8])

        writer.write_frame(img)
    writer.close()
    reader.close()


def render_global(cfg, retarget=False):
    global_video_path = Path(cfg.paths.global_video)
    if global_video_path.exists():
        Log.info(f"[Render Global] Video already exists at {global_video_path}")
        return

    debug_cam = False
    pred = torch.load(cfg.paths.hmr4d_results)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()

    # smpl
    global_transl = pred["smpl_params_global"]["transl"]
    incam_transl_start = pred["smpl_params_incam"]["transl"][:, 0]
    if retarget:
        pred["smpl_params_global"]["transl"] = retarget_transl(global_transl, incam_transl_start, xz_only=False)

    person_num = pred["smpl_params_global"]["transl"].shape[0]
    merged_verts = []
    for person_idx in range(person_num):
        smplx_out = smplx(**to_cuda(fetch_smpl_params(pred["smpl_params_global"], person_idx)))
        pred_ay_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])
        merged_verts.append(pred_ay_verts)
    pred_ay_verts = torch.stack(merged_verts, dim=1)  # (F, P, V, 3)
    
    # position
    all_offset = []
    for person_idx in range(person_num):
        verts = pred_ay_verts[:, person_idx].clone()  # (L, V, 3)
        offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # (3)
        offset[1] = verts[:, :, [1]].min()
        all_offset.append(offset)
    offset = torch.mean(torch.stack(all_offset, dim=0), dim=0)
    
    # pred_ay_verts = pred_ay_verts - offset
    # T_ay2ayfz = compute_T_ayfz2ay(einsum(J_regressor, pred_ay_verts[[0], 0], "j v, l v i -> l j i"), inverse=True)
    
    # verts_glob = []
    # for person_idx in range(person_num):    
    #     # verts_glob_person = apply_T_on_points(pred_ay_verts[:, person_idx], T_ay2ayfz)
    #     verts_glob_person = torch.einsum("...ki,...ji->...jk", T_ay2ayfz[..., :3, :3], pred_ay_verts[:, person_idx])
    #     verts_glob.append(verts_glob_person)
    # verts_glob = torch.stack(verts_glob, dim=1)  # (F, P, V, 3)
    
    verts_glob = pred_ay_verts - offset
        
    joints_glob = einsum(J_regressor, verts_glob[:, 0], "j v, l v i -> l j i")  # (L, J, 3)
    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob[:, 0].cpu(),
        beta=3.0,
        cam_height_degree=25,
        target_center_height=1.0,
    )
    # -- rendering code -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    _, _, K = create_camera_sensor(width, height, 18)  # render as 24mm lens

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob[:, 0])
    renderer.set_ground(scale * 4.0, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    render_length = length if not debug_cam else 8
    writer = get_writer(global_video_path, fps=30, crf=CRF)
    for i in tqdm(range(render_length), desc=f"Rendering Global"):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(verts_glob[i], color[None].repeat(person_num, 1), cameras, global_lights)
        writer.write_frame(img)
    writer.close()


if __name__ == "__main__":
    cfg = parse_args_to_cfg()
    paths = cfg.paths
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f'[GPU]: {torch.cuda.get_device_properties("cuda")}')

    # ===== Preprocess and save to disk ===== #
    run_preprocess(cfg)
    data = load_data_dict(cfg)

    # ===== HMR4D ===== #
    if not Path(paths.hmr4d_results).exists():
        Log.info("[HMR4D] Predicting")
        model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        tic = Log.sync_time()
        pred = model.predict_multiperson(data, static_cam=cfg.static_cam)
        pred = detach_to_cpu(pred)
        data_time = data["length"] / 30
        Log.info(f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s for data-length={data_time:.1f}s")
        torch.save(pred, paths.hmr4d_results)

    # ===== Render ===== #
    render_incam(cfg)
    render_global(cfg, retarget=True)
    if not Path(paths.incam_global_horiz_video).exists():
        Log.info("[Merge Videos]")
        merge_videos_horizontal([paths.incam_video, paths.global_video], paths.incam_global_horiz_video)

"""
CUDA_VISIBLE_DEVICES=1, python -m tools.demo.demo_multiperson --video=/mnt/data/jing/Video_Generation/video_data_repos/video_preprocessor/WHAM/examples/two_persons.mp4 --output_root outputs/demo_mp --recreate_video
"""
