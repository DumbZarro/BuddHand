import os
import time
import warnings

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from loguru import logger
from mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model, vis_pose_tracking_result)
from mmpose.datasets import DatasetInfo
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess

# camera config
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

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

# pose config
det_config = "../model/mmpose/faster_rcnn_r50_fpn_coco.py"
# det_checkpoint = "../model/mmpose/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
det_checkpoint = "../model/weight/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
pose_config = "../model/mmpose/mobilenetv2_onehand10k_256x256.py"
# pose_checkpoint = "../model/mmpose/mobilenetv2_onehand10k_256x256.pth"
pose_checkpoint = "../model/weight/mobilenetv2_onehand10k_256x256.pth"
video_path = "D:\\Code\\PythonCode\\YOLOX\\assets\\hand5.mp4"
pose_device = 'cuda:0'
out_video_root = ""
det_cat_id = 1  # Category id for bounding box detection model
bbox_thr = 0.3  # Bounding box score threshold
kpt_thr = 0.3  # Keypoint score threshold
use_oks_tracking = False  # Using OKS tracking
tracking_thr = 0.3  # Tracking threshold
euro = False  # Using One_Euro_Filter for smoothing
radius = 4  # Keypoint radius for visualization
thickness = 1  # Link thickness for visualization


# 获取深度图像
def getDeepMap():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    # depth_data = np.asanyarray(aligned_depth_frame.get_data(), dtype="float16")
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_mapped_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    # return color_image, depth_mapped_image
    return color_image, aligned_depth_frame, depth_mapped_image


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
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def main(exp):
    ################################## detect ##############################################
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

    ####################################### pose #########################################

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        pose_config,
        pose_checkpoint,
        device=pose_device.lower()
    )

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    fps = None
    next_id = 0
    pose_results = []

    ####################################### start #########################################

    while True:

        # timer
        start_t= time.time()

        pose_results_last = pose_results
        # camera
        img, aligned_depth_frame, depth_mapped_image = getDeepMap()

        # detect hand
        outputs, img_info = yolox.inference(img)
        if outputs == [None]:
            continue
        # print(outputs)
        # print(outputs[0])
        # print(outputs[0].cpu().numpy().tolist())
        blist = outputs[0].cpu().numpy().tolist()
        person_results = []
        for item in blist:
            person_results.append(dict(bbox=np.array(item[:5], dtype=np.float32)))

        # hand pose (test a single image, with a list of bboxes)
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=False,  # dosen't work
            outputs=None)  # e.g. use ('backbone', ) to return backbone feature ? dosen't work

        # get track id for each person instance
        pose_results, next_id = get_track_id(
            pose_results,
            pose_results_last,
            next_id,
            use_oks=use_oks_tracking,
            tracking_thr=tracking_thr,
            use_one_euro=euro,
            fps=fps)

        # show the results
        vis_img = vis_pose_tracking_result(
            pose_model,
            img,
            pose_results,
            radius=radius,
            thickness=thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=kpt_thr,
            show=False)

        # cv2.imshow('Image', vis_img)

        # timer
        all_time = time.time()-start_t
        print(all_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    exp = get_exp(exp_file, name)
    main(exp)
