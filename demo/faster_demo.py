import time

import cv2
import numpy as np
import pyrealsense2 as rs

from utils.Ascender import roi_generator_Ascender
from utils.CameraConfig import pipeline
from utils.Normalizer import roi_normalizer
from utils.PoseConfig import get_trackid, tracking_result, pose_inference
from utils.Render import show_3d_blocking,show_3d
from utils.Selector import select_stream

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()  # decimate 毁灭
decimate.set_option(rs.option.filter_magnitude, 2)
colorizer = rs.colorizer()


def get_stream():
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame()
    # depth_frame = decimate.process(depth_frame)   # 注意!这里缩小了一半
    color_frame = frames.get_color_frame()

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


def main():
    fps = None
    next_id = 0
    pose_results = []

    flag = False  # 上一帧是否检测到手部关键点,初始化为否
    MBR = []

    while True:

        # timer
        start_t = time.time()

        # tracking
        pose_results_last = pose_results

        # camera
        depth_image, color_image = get_stream()

        # 选择器 输出流
        flag, hand_results = select_stream(flag, color_image, MBR)
        if flag == False:
            continue  # 没有检测到手部,直接跳到下一帧,并且调用目标检测器

        # 标准化 加大力度
        normal_results = roi_normalizer(hand_results, 10)

        # 关键点检测
        # hand pose (test a single image, with a list of bboxes)
        pose_results, returned_outputs = pose_inference(color_image, normal_results)


        # 计算各个点的置信度平均值,如果检测出的点低于 30% (21/3=7) 则视为检测失败从新调用
        # if flag and np.sum(pose_results[0]["keypoints"], axis=0)[2] < 7:
        #     flag = False  # 没有检测到手势,下一帧要调用目标检测.
        #     MBR = []
        #     continue
        if flag and np.sum(pose_results, axis=0)[2] < 7:
            flag = False  # 没有检测到手势,下一帧要调用目标检测.
            MBR = []
            continue

        if  pose_results==[]:
            flag = False  # 没有检测到手势,下一帧要调用目标检测.
            MBR = []
            continue

        # TODO 处理MBR 和 hand_result
        MBR, xyz = roi_generator_Ascender(pose_results, depth_image)
        # print(xyz)

        # 目标跟踪,获取id
        pose_results, next_id = get_trackid(pose_results, pose_results_last, next_id, fps)

        # timer
        all_time = time.time() - start_t
        # print(all_time)

        # 渲染 show the results
        vis_img = tracking_result(color_image, pose_results)

        # show_3d_blocking(xyz)
        # show_3d(xyz)

        # print(len(pose_results[0]["keypoints"]))
        # print(pose_results[0]["keypoints"])

        cv2.imshow('Image', vis_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
