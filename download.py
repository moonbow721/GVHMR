import os
import gdown

download_path = './inputs/checkpoints/'

os.makedirs(download_path, exist_ok=True)

files = [
    ("dpvo/dpvo.pth", "1DE5GVftRCfZOTMp8YWF0xkGudDxK0nr0"),
    ("gvhmr/gvhmr_siga24_release.ckpt", "1c9iCeKFN4Kr6cMPJ9Ss6Jdc3SZFnO5NP"),
    ("hmr2/epoch=10-step=25000.ckpt", "1X5hvVqvqI9tvjUCb2oAlZxtgIKD9kvsc"),
    ("vitpose/vitpose-h-multi-coco.pth", "1sR8xZD9wrZczdDVo6zKscNLwvarIRhP5"),
    ("yolo/yolov8x.pt", "1_HGm-lqIH83-M1ML4bAXaqhm_eT2FKo5")
]

for file_path, file_id in files:
    target_path = os.path.join(download_path, file_path)
    
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    gdown.download(f'https://drive.google.com/uc?id={file_id}', target_path, quiet=False)

print("Download completed!")
