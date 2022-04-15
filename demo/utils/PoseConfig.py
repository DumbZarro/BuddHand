import warnings

from mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model, vis_pose_tracking_result, vis_3d_pose_result)
from mmpose.datasets import DatasetInfo

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


def get_pose():
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

    return pose_model, dataset, dataset_info


pose_model, dataset, dataset_info = get_pose()


def pose_inference(color_image, normal_results):
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
    return pose_results, returned_outputs


def get_trackid(pose_results, pose_results_last, next_id, fps):
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
    return pose_results, next_id


def tracking_result(color_image, pose_results):
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
    return vis_img


def vis_3d(pose_results_vis, color_image):
    vis_3d_pose_result(
        pose_model,
        result=pose_results_vis,  # 关键点
        img=color_image,
        out_file="out_file",
        # dataset=dataset, # 'InterHand3DDataset' 'Body3DH36MDataset'
        show=True,
        kpt_score_thr=kpt_thr,
        radius=radius,
        thickness=thickness,
    )
