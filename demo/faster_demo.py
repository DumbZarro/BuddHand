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

profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height
print("w")
print(w)
print("h")
print(h)

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()  # decimate 毁灭
decimate.set_option(rs.option.filter_magnitude, 2)
colorizer = rs.colorizer()

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
det_checkpoint = "../model/mmpose/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
pose_config = "../model/mmpose/mobilenetv2_onehand10k_256x256.py"
pose_checkpoint = "../model/mmpose/mobilenetv2_onehand10k_256x256.pth"
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
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def get_stream():
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame()
    # depth_frame = decimate.process(depth_frame)   # 注意!这里缩小了一半
    color_frame = frames.get_color_frame()

    # global w, h
    # w, h = depth_intrinsics.width, depth_intrinsics.height
    # print("w")
    # print(w)
    # print("h")
    # print(h)

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # print("depth_image.shape")
    # print(depth_image.shape)  # 注意!宽度和深度是反过来的
    # print("depth_image")
    # print(depth_image)

    # mapped_frame, color_source = color_frame, color_image
    #
    # points = pc.calculate(depth_frame)
    # pc.map_to(mapped_frame)
    #
    # # Pointcloud data to arrays
    # v, t = points.get_vertices(), points.get_texture_coordinates()  # get_vertices 顶点 get_texture_coordinates 贴图坐标
    # verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz     空间坐标
    # texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv  投影坐标
    return depth_image, color_image


def select_stream(flag, yolox, color_image, MBR):
    if flag == True:  # 上一帧有检测到手势,使用上一帧的最小外接矩形
        return True, MBR  # 同时默认这一帧也能检测到,检测不到后续再给False
    else:
        # detect hand
        hand_results = detect_hand(yolox, color_image)
        if hand_results == []:
            return False, []
        else:
            return True, hand_results  # 检测到手势


# 输入各个姿态的点,以及需要扩大的距离
def roi_normalizer(hand_results, dist):
    # 逐个扩大并限制高度
    for item in hand_results:
        # print("roi_normalizer item")
        # print(item)
        item[0] = max((int(item["bbox"][0]) - dist), 0)  # x
        item[1] = max((int(item["bbox"][1]) - dist), 0)  # y
        item[2] = min((int(item["bbox"][2]) + 2 * dist), w)  # w
        item[3] = min((int(item["bbox"][3]) + 2 * dist), h)  # h

    return hand_results


# TODO 为什么x轴算出来的都输负数
# The cameras principle point are descried by ppx and ppy,
# thus correspond to cx and cy respectively which is usally used in literature.
def depth2xyz(u, v, depth, fx, fy, ppx, ppy):
    # depth = depth * 0.001  # depth2mi
    # 有畸变 ppx 和 ppy 是用来j矫正的参数

    z = float(depth)
    x = float((u - ppx) * z) / fx
    y = float((v - ppy) * z) / fy

    result = [x, y, z]
    return result


def roi_generator_Ascender(pose_results, depth_image):
    min_x = w
    min_y = h
    max_x = 0
    max_y = 0

    # pose_results[0]["keypoints"] 这个会outof range
    for item in pose_results[0]["keypoints"]:  # 每个item前两个是关键点的坐标,后面一个置信度.
        # print("pose_results[0][keypoints] item")
        # print(item)
        # 最小外接矩形
        if item[0] < min_x:
            min_x = item[0]
        if item[0] > max_x:
            max_x = item[0]
        if item[1] < min_y:
            min_y = item[1]
        if item[1] > max_y:
            max_y = item[1]

        # 映射 升维
        x = max(min(int(item[0]), h - 1), 0)
        y = max(min(int(item[1]), w - 1), 0)
        xyz = depth2xyz(item[0], item[1],
                        depth_image[x, y],  # 宽高是相反的,heatmap预测的是浮点要转int,同时要从零开始
                        depth_intrinsics.fx,
                        depth_intrinsics.fy,
                        depth_intrinsics.ppx,
                        depth_intrinsics.ppy)
        # print("xyz")
        # print(xyz)

    # print("min_x")
    # print(min_x)
    # print("min_y")
    # print(min_y)
    # print("max_x")
    # print(max_x)
    # print("max_y")
    # print(max_y)
    bbox = [min_x, min_y, max_x, max_y, 0.99]  # Minimum Area Bounding Rectangle 最后面一个是置信度
    MBR = [dict(bbox=np.array(bbox, dtype=np.float32))]

    # print("MBR")
    # print(MBR)
    return MBR, xyz


def detect_hand(yolox, color_image):
    outputs, img_info = yolox.inference(color_image)
    hand_results = []
    if outputs == [None]:
        return hand_results
    # print(outputs)
    # print(outputs[0])
    # print(outputs[0].cpu().numpy().tolist())
    bbox_list = outputs[0].cpu().numpy().tolist()
    for item in bbox_list:
        hand_results.append(dict(bbox=np.array(item[:5], dtype=np.float32)))
    # print("hand_results:")
    # print(hand_results)

    return hand_results


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

    flag = False  # 上一帧是否检测到手部关键点,初始化为否
    MBR = []

    while True:

        # tracking
        pose_results_last = pose_results

        # camera
        depth_image, color_image = get_stream()

        # TODO 选择器 输出流

        flag, hand_results = select_stream(flag, yolox, color_image, MBR)
        if flag == False:
            continue  # 没有检测到手部,直接跳到下一帧,并且调用目标检测器

        # 标准化 加大力度
        normal_results = roi_normalizer(hand_results, 10)

        # 关键点检测
        # hand pose (test a single image, with a list of bboxes)
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            color_image,
            normal_results,
            bbox_thr=bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=True,
            outputs=None)

        # 升维 获得最小外接矩形
        # 大意了,pose result返回的坐标是浮点(因为是heatmap)和像素无关 但是官方的是直接加一个int就行了啊
        # print("pose_results")
        # print(pose_results)
        if pose_results == []:
            flag = False  # 没有检测到手势,下一帧要调用目标检测.
            continue
        # print("=======")
        # print(returned_outputs)
        # print("pose_results[0]")
        # print(pose_results[0])
        # print(pose_results[0].keys())
        # print(pose_results[0]["keypoints"])

        # TODO 处理MBR 和 hand_result
        MBR, xyz = roi_generator_Ascender(pose_results, depth_image)

        # 目标跟踪,获取id
        # get track id for each person instance
        pose_results, next_id = get_track_id(
            pose_results,
            pose_results_last,
            next_id,
            use_oks=use_oks_tracking,
            tracking_thr=tracking_thr,
            use_one_euro=euro,
            fps=fps)

        # print(pose_results)   # 和上面差不多,多了area和track_id
        # print("pose_results")

        # 渲染
        # show the results
        vis_img = vis_pose_tracking_result(
            pose_model,
            color_image,
            pose_results,
            radius=radius,
            thickness=thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=kpt_thr,
            show=False)

        cv2.imshow('Image', vis_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    exp = get_exp(exp_file, name)
    main(exp)
