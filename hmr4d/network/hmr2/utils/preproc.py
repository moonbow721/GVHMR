import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225])


def expand_to_aspect_ratio(input_shape, target_aspect_ratio=[192, 256]):
    """Increase the size of the bounding box to match the target shape."""
    if target_aspect_ratio is None:
        return input_shape

    try:
        w, h = input_shape
    except (ValueError, TypeError):
        return input_shape

    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        h_new = max(w * h_t / w_t, h)
        w_new = w
    else:
        h_new = h
        w_new = max(h * w_t / h_t, w)
    if h_new < h or w_new < w:
        breakpoint()
    return np.array([w_new, h_new])


def crop_and_resize(img, bbx_xy, bbx_s, dst_size=256, enlarge_ratio=1.2):
    """
    Args:
        img: (H, W, 3), [0, 255]
        bbx_xy: (2,)
        bbx_s: scalar
    
    Returns:
        img_crop: (dst_size, dst_size, 3)
        bbx_xys_final: (3,)
    """
    hs = bbx_s * enlarge_ratio / 2
    src = np.stack(
        [
            bbx_xy - hs,  # left-up corner
            bbx_xy + np.array([hs, -hs]),  # right-up corner
            bbx_xy,  # center
        ]
    ).astype(np.float32)
    dst = np.array([[0, 0], [dst_size - 1, 0], [dst_size / 2 - 0.5, dst_size / 2 - 0.5]], dtype=np.float32)
    A = cv2.getAffineTransform(src, dst)
    img_crop = cv2.warpAffine(img, A, (dst_size, dst_size), flags=cv2.INTER_LINEAR)
    bbx_xys_final = np.array([*bbx_xy, bbx_s * enlarge_ratio])
    return img_crop, bbx_xys_final

def crop_and_resize_torch(img, bbx_xy, bbx_s, dst_size=256, enlarge_ratio=1.2):
    """
    Args:
        img: (B, C, H, W) tensor, [0, 255]
        bbx_xy: (B, 2) tensor
        bbx_s: (B,) tensor
        
    Returns:
        img_crop: (B, C, dst_size, dst_size) tensor
        bbx_xys_final: (B, 3) tensor
    """
    batch_size = img.shape[0]
    device = img.device

    hs = bbx_s[:, None] * enlarge_ratio / 2
    src = torch.stack([bbx_xy - hs, bbx_xy + torch.cat([hs, -hs], dim=1), bbx_xy], dim=1)

    dst = torch.tensor([[0, 0], [dst_size - 1, 0], [dst_size / 2 - 0.5, dst_size / 2 - 0.5]], 
                       dtype=torch.float32, device=device).repeat(batch_size, 1, 1)

    # Compute the affine transformation matrix
    scale_x = (dst[:, 1, 0] - dst[:, 0, 0]) / (src[:, 1, 0] - src[:, 0, 0])
    scale_y = (dst[:, 2, 1] - dst[:, 0, 1]) / (src[:, 2, 1] - src[:, 0, 1])
    
    # Translation values
    trans_x = dst[:, 0, 0] - scale_x * src[:, 0, 0]
    trans_y = dst[:, 0, 1] - scale_y * src[:, 0, 1]

    A = torch.zeros(batch_size, 3, 3).to(device)
    A[:, 0, 0] = scale_x
    A[:, 0, 2] = trans_x
    A[:, 1, 1] = scale_y
    A[:, 1, 2] = trans_y
    A[:, 2, 2] = 1

    M = transform_affine_matrix(A, img.shape[2], img.shape[3], dst_size, dst_size)
    img_crop = warp_perspective(img, M, (dst_size, dst_size), mode='bilinear', padding_mode='zeros')

    bbx_xys_final = torch.cat([bbx_xy, bbx_s[:, None] * enlarge_ratio], dim=1)
    return img_crop, bbx_xys_final


def transform_affine_matrix(M, src_h, src_w, dst_h, dst_w):
    batch_size = M.shape[0]
    device = M.device
    r = torch.tensor(
        [
            [2.0 / src_w, 0, -1],
            [0, 2.0 / src_h, -1],
            [0, 0, 1]
        ],
        dtype=torch.float32, device=device
    ).unsqueeze(0).repeat(batch_size, 1, 1)

    t_ = torch.tensor(
        [
            [dst_w / 2.0, 0, dst_w / 2.0],
            [0, dst_h / 2.0, dst_h / 2.0],
            [0, 0, 1]
        ],
        dtype=torch.float32, device=device
    ).unsqueeze(0).repeat(batch_size, 1, 1)

    theta = torch.bmm(torch.bmm(r, torch.inverse(M)), t_)

    return theta


def warp_perspective(src, M, dsize, mode='bilinear', padding_mode='zeros'):
    assert src is not None and M is not None
    assert len(src.shape) == 4  # (B, C, H, W)
    assert M.shape[1:] == (3, 3)

    B, C, H, W = src.shape
    dst_height, dst_width = dsize

    M_trans = M[:, :2, :]
    grid = F.affine_grid(M_trans, [B, C, dst_height, dst_width], align_corners=False)
    dst = F.grid_sample(src, grid, mode=mode, padding_mode=padding_mode, align_corners=False)

    return dst
