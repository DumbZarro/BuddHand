import numpy as np

from .CameraConfig import depth_intrinsics
from .PoseConfig import kpt_thr

w, h = depth_intrinsics.width, depth_intrinsics.height


# TODO 为什么x轴算出来的都输负数
# The cameras principle point are descried by ppx and ppy,
# thus correspond to cx and cy respectively which is usally used in literature.
def depth2xyz(u, v, depth, fx, fy, ppx, ppy):
    # depth = depth * 0.001  # depth2mi
    # 有畸变 ppx 和 ppy 是用来j矫正的参数

    # TODO 版本1好像不太正确
    z = float(depth)
    x = float((u - ppx) * z) / fx
    y = float((v - ppy) * z) / fy
    result = [x, y, z]

    # 版本2
    # result = [u, v, depth]

    return result


# 2Dto3D
def roi_generator_Ascender(pose_results, depth_image):
    min_x = w
    min_y = h
    max_x = 0
    max_y = 0
    xyz = []
    # pose_results[0]["keypoints"] 这个会outof range
    for item in pose_results[0]["keypoints"]:  # 每个item前两个是关键点的坐标,后面一个置信度.
        # print("pose_results[0][keypoints] item")
        # print(item)

        if item[2] < kpt_thr: continue

        # 计算最小外接矩形
        if item[0] < min_x: min_x = item[0]
        if item[0] > max_x: max_x = item[0]
        if item[1] < min_y: min_y = item[1]
        if item[1] > max_y: max_y = item[1]

        # 映射 升维
        # 计算坐标
        x = max(min(int(item[0]), h - 1), 0)  # 宽高是相反的,heatmap预测的是浮点要转int,同时要从零开始
        y = max(min(int(item[1]), w - 1), 0)
        res = depth2xyz(item[0], item[1],
                        depth_image[x, y],
                        depth_intrinsics.fx,
                        depth_intrinsics.fy,
                        depth_intrinsics.ppx,
                        depth_intrinsics.ppy)
        xyz.append(res)
        # print("xyz"+xyz)

    bbox = [min_x, min_y, max_x, max_y, 0.99]  # Minimum Area Bounding Rectangle 最后面一个是置信度
    MBR = [dict(bbox=np.array(bbox, dtype=np.float32))]
    # print("MBR"+MBR)
    return MBR, xyz
