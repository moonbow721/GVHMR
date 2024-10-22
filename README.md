# GVHMR: World-Grounded Human Motion Recovery via Gravity-View Coordinates
### [Project Page](https://zju3dv.github.io/gvhmr) | [Paper](https://arxiv.org/abs/2409.06662)

> World-Grounded Human Motion Recovery via Gravity-View Coordinates  
> [Zehong Shen](https://zehongs.github.io/)<sup>\*</sup>,
[Huaijin Pi](https://phj128.github.io/)<sup>\*</sup>,
[Yan Xia](https://isshikihugh.github.io/scholar),
[Zhi Cen](https://scholar.google.com/citations?user=Xyy-uFMAAAAJ),
[Sida Peng](https://pengsida.net/)<sup>†</sup>,
[Zechen Hu](https://zju3dv.github.io/gvhmr),
[Hujun Bao](http://www.cad.zju.edu.cn/home/bao/),
[Ruizhen Hu](https://csse.szu.edu.cn/staff/ruizhenhu/),
[Xiaowei Zhou](https://xzhou.me/)  
> SIGGRAPH Asia 2024

<p align="center">
    <img src=docs/example_video/project_teaser.gif alt="animated" />
</p>

## Setup

Please see [installation](docs/INSTALL.md) for details.
Don't install this repo as a package, it will cause errors when importing other modules.

Install [hamer](https://github.com/geopavlakos/hamer) and link the vitpose-wholebody checkpoint in hamer (`./_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth`) to this repo `./inputs/checkpoints/vitpose/vitpose-h-coco-wholebody.pth`.

## Quick Start

### [<img src="https://i.imgur.com/QCojoJk.png" width="30"> Google Colab demo for GVHMR](https://colab.research.google.com/drive/1N9WSchizHv2bfQqkE9Wuiegw_OT7mtGj?usp=sharing)

### [<img src="https://s2.loli.net/2024/09/15/aw3rElfQAsOkNCn.png" width="20"> HuggingFace demo for GVHMR](https://huggingface.co/spaces/LittleFrog/GVHMR)

### Demo
Demo entries are provided in `tools/demo`. Use `-s` to skip visual odometry if you know the camera is static, otherwise the camera will be estimated by DPVO.
We also provide a script `demo_folder.py` to inference a entire folder.
```shell
python -m tools.demo.demo --video=docs/example_video/tennis.mp4 -s
python -m tools.demo.demo_folder -f inputs/demo/folder_in -d outputs/demo/folder_out -s
python -m tools.demo.demo_multiperson --video=docs/example_video/two_persons.mp4 --output_root outputs/demo_mp --batch_size 64 --export_npy
python -m tools.demo.demo_multiperson --video=docs/example_video/vertical_dance.mp4 --output_root outputs/demo_mp -s
```

### Reproduce
1. **Test**:
To reproduce the 3DPW, RICH, and EMDB results in a single run, use the following command:
    ```shell
    python tools/train.py global/task=gvhmr/test_3dpw_emdb_rich exp=gvhmr/mixed/mixed ckpt_path=inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt
    ```
    To test individual datasets, change `global/task` to `gvhmr/test_3dpw`, `gvhmr/test_rich`, or `gvhmr/test_emdb`.

2. **Train**:
To train the model, use the following command:
    ```shell
    # The gvhmr_siga24_release.ckpt is trained with 2x4090 for 420 epochs, note that different GPU settings may lead to different results.
    python tools/train.py exp=gvhmr/mixed/mixed
    ```
    During training, note that we do not employ post-processing as in the test script, so the global metrics results will differ (but should still be good for comparison with baseline methods).

# Different from the original repo

This version of the repository includes modifications to support multi-person HMR:

1. Multi-person tracking:
   - Updated the `Tracker` class to return bounding boxes for multiple people using `get_all_tracks` instead of `get_one_track`.
   - Modified preprocessing to handle multiple person detections and features.

2. Multi-person pose estimation:
   - Adapted the `VitPoseExtractor` to process multiple people simultaneously.
   - Updated the feature extraction process to handle batches of multiple people.

3. Multi-person SMPL reconstruction:
   - Modified the `DemoPL` class to predict SMPL parameters for multiple people.
   - Updated the rendering process to handle multiple SMPL models in both in-camera and global coordinate systems.

4. Rendering improvements:
   - Implemented merged faces creation for rendering multiple SMPL models simultaneously.
   - Added support for retargeting global translations to better align with in-camera positions.

5. New demo script:
   - Added `demo_multiperson.py` to showcase the multi-person reconstruction pipeline.
   - Includes options for batch processing and verbose output for debugging.

6. Performance optimizations:
   - Introduced batch processing for VitPose and feature extraction to improve efficiency.

# Results format

## Preprocessing results

1. `/preprocess/bbx.pt`:
   - Contains bounding box information for multiple people
   - `bbx_xyxy`: Tensor of shape (P, L, 4), where P is the number of people and L is the number of frames
   - `bbx_xys`: Tensor of shape (P, L, 3), containing center coordinates and scale for each bounding box

2. `/preprocess/slam_results.pt`:
   - Camera pose estimation results (if not using static camera)
   - NumPy array of shape (L, 7), where each row contains [x, y, z, qx, qy, qz, qw]

3. `/preprocess/vitpose.pt`:
   - 2D pose estimation results
   - Tensor of shape (P, L, 17, 3), where 17 is the number of keypoints and 3 represents [x, y, confidence]

4. `/preprocess/vit_features.pt`:
   - Image features extracted from the video frames
   - Tensor of shape (P, L, 1024), where 1024 is the feature dimension

## GVHMR reconstruction results

The main reconstruction results are stored in `hmr4d_results.pt`, which contains the following keys:

1. `smpl_params_global` and `smpl_params_incam`:
   - SMPL parameters for global and in-camera coordinate systems
   - Each contains:
     - `body_pose`: Tensor of shape (P, L, 63)
     - `betas`: Tensor of shape (P, L, 10)
     - `global_orient`: Tensor of shape (P, L, 3)
     - `transl`: Tensor of shape (P, L, 3)

2. `K_fullimg`:
   - Camera intrinsic matrix
   - Tensor of shape (L, 3, 3), same across all frames

3. `net_outputs`:
   - Additional network outputs (not used for now)

## Warning

The `smpl_params_global` params of different people starts from the same origin. To visualize the results, I retarget the global translations based on the first-frame of `smpl_params_incam` params.

# Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{shen2024gvhmr,
  title={World-Grounded Human Motion Recovery via Gravity-View Coordinates},
  author={Shen, Zehong and Pi, Huaijin and Xia, Yan and Cen, Zhi and Peng, Sida and Hu, Zechen and Bao, Hujun and Hu, Ruizhen and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia Conference Proceedings},
  year={2024}
}
```

# Acknowledgement

We thank the authors of
[WHAM](https://github.com/yohanshin/WHAM),
[4D-Humans](https://github.com/shubham-goel/4D-Humans),
and [ViTPose-Pytorch](https://github.com/gpastal24/ViTPose-Pytorch) for their great works, without which our project/code would not be possible.
