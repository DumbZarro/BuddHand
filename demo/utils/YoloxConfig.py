import os

import cv2
import torch
from loguru import logger
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess

# detect config
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
conf = 0.25
nms = 0.3
demo = "webcam"
camid = 0
exp_file = "../model/YOLOX/yolox_s.py"
ckpt_file = "../model/YOLOX/v1_conf03_nms045.pth"
det_device = 'gpu'
save_result = True
tsize = 640
fp16 = False  # Adopting mix precision evaluating.
legacy = False  # To be compatible with older versions
fuse = True  # Fuse conv and bn for testing 当前CNN卷积层的基本组成单元标配：Conv + BN +ReLU 三剑客。但其实在网络的推理阶段，可以将BN层的运算融合到Conv层中，减少运算量，加速推理。
path = "./assets/dog.jpg"
experiment_name = None
name = None
exp = get_exp(exp_file, name)


class YOLOXPredictor(object):
    def __init__(
            self,
            model,
            exp,
            cls_names=COCO_CLASSES,
            decoder=None,
            device="cpu",
            legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.preproc = ValTransform(legacy=legacy)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        if self.device == "gpu":
            img = img.cuda()

        with torch.no_grad():
            # t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def get_yolox():
    global experiment_name
    if not experiment_name:
        experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if conf is not None:
        exp.test_conf = conf
    if nms is not None:
        exp.nmsthre = nms
    if tsize is not None:
        exp.test_size = (tsize, tsize)

    det_model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(det_model, exp.test_size)))

    if det_device == "gpu":
        det_model.cuda()
    det_model.eval()

    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    det_model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    if fuse:
        logger.info("\tFusing model...")
        det_model = fuse_model(det_model)

    decoder = None
    yolox = YOLOXPredictor(det_model, exp, COCO_CLASSES, decoder, det_device, legacy)
    return yolox


yolox = get_yolox()
