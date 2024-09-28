import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from hmr4d.utils.pylogger import Log
from hmr4d.configs import MainStore, builds

from hmr4d.utils.geo.hmr_cam import normalize_kp2d


class DemoPL(pl.LightningModule):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)

    @torch.no_grad()
    def predict(self, data, static_cam=False):
        """auto add batch dim
        data: {
            "length": int, or Torch.Tensor,
            "kp2d": (F, 17, 3)
            "bbx_xys": (F, 3)
            "K_fullimg": (F, 3, 3)
            "cam_angvel": (F, 6)
            "f_imgseq": (F, 1024)
        }

        """
        # ROPE inference
        batch = {
            "length": data["length"][None],
            "obs": normalize_kp2d(data["kp2d"], data["bbx_xys"])[None],
            "bbx_xys": data["bbx_xys"][None],
            "K_fullimg": data["K_fullimg"][None],
            "cam_angvel": data["cam_angvel"][None],
            "f_imgseq": data["f_imgseq"][None],
        }
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = self.pipeline.forward(batch, train=False, postproc=True, static_cam=static_cam)

        pred = {
            "smpl_params_global": {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()},
            "smpl_params_incam": {k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()},
            "K_fullimg": data["K_fullimg"],
            "net_outputs": outputs,  # intermediate outputs
        }
        return pred

    @torch.no_grad()
    def predict_multiperson(self, data, static_cam=False):
        """auto add batch dim
        data: {
            "length": int, or Torch.Tensor,
            "kp2d": (num_person, F, 17, 3)
            "bbx_xys": (num_person, F, 3)
            "K_fullimg": (F, 3, 3)
            "cam_angvel": (F, 6)
            "f_imgseq": (num_person, F, 1024)
        }

        """
        # ROPE inference
        num_person = data["kp2d"].shape[0]
        batch = {
            "length": data["length"].repeat(num_person,),
            "obs": normalize_kp2d(data["kp2d"], data["bbx_xys"]),
            "bbx_xys": data["bbx_xys"],
            "K_fullimg": data["K_fullimg"][None].repeat(num_person, 1, 1, 1),
            "cam_angvel": data["cam_angvel"][None].repeat(num_person, 1, 1),
            "f_imgseq": data["f_imgseq"],
        }
        # import ipdb; ipdb.set_trace()
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = self.pipeline.forward(batch, train=False, postproc=True, static_cam=static_cam)

        pred = {
            "smpl_params_global": {k: v for k, v in outputs["pred_smpl_params_global"].items()},
            "smpl_params_incam": {k: v for k, v in outputs["pred_smpl_params_incam"].items()},
            "K_fullimg": data["K_fullimg"],
            "net_outputs": outputs,  # intermediate outputs
        }
        return pred
    
    def load_pretrained_model(self, ckpt_path):
        """Load pretrained checkpoint, and assign each weight to the corresponding part."""
        Log.info(f"[PL-Trainer] Loading ckpt type: {ckpt_path}")

        state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            Log.warn(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            Log.warn(f"Unexpected keys: {unexpected}")


MainStore.store(name="gvhmr_pl_demo", node=builds(DemoPL, pipeline="${pipeline}"), group="model/gvhmr")
