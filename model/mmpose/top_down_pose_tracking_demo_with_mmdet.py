# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings

import cv2
from mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_tracking_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# 初始化参数
det_config = "./faster_rcnn_r50_fpn_coco.py"
det_checkpoint = "./faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
pose_config = "./mobilenetv2_onehand10k_256x256.py"
pose_checkpoint = "./mobilenetv2_onehand10k_256x256.pth"
video_path = "D:\\Code\\PythonCode\\YOLOX\\assets\\hand5.mp4"
device = 'cuda:0'
out_video_root = ""
det_cat_id = 1  # Category id for bounding box detection model
bbox_thr = 0.3  # Bounding box score threshold
kpt_thr = 0.3  # Keypoint score threshold
use_oks_tracking = False  # Using OKS tracking
tracking_thr = 0.3  # Tracking threshold
euro = False  # Using One_Euro_Filter for smoothing
radius = 4  # Keypoint radius for visualization
thickness = 1  # Link thickness for visualization


def main():
    # 检测器的 config checkpoint
    det_model = init_detector(
        det_config,
        det_checkpoint,
        device=device.lower()
    )
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        pose_config,
        pose_checkpoint,
        device=device.lower()
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

    cap = cv2.VideoCapture(video_path)
    fps = None

    assert cap.isOpened(), f'Faild to load video file {video_path}'

    if out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(out_video_root,
                         f'vis_{os.path.basename(video_path)}'), fourcc,
            fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    next_id = 0
    pose_results = []
    while (cap.isOpened()):
        pose_results_last = pose_results

        flag, img = cap.read()
        if not flag:
            break
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, img)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, det_cat_id)

        print(person_results)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

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

        cv2.imshow('Image', vis_img)

        # if save_out_video:
        #     videoWriter.write(vis_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save_out_video:
        videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()