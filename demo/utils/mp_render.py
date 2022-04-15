import math
import cv2
import mediapipe as mp
from .CameraConfig import w,h

# 导入solution
mp_hands = mp.solutions.hands

# 导入绘图函数
mpDraw = mp.solutions.drawing_utils

def get_distance(p1, p2):
    return math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2)


def draw_hand(xyz,img):
    # 获取该手的21个关键点坐标
    hand_21 = xyz

    # print(hand_21)

    # 可视化关键点及骨架连线
    mpDraw.draw_landmarks(img, hand_21, mp_hands.HAND_CONNECTIONS)

    # 记录左右手信息

    # 获取手腕根部深度坐标
    finesse_z = hand_21.landmark[0].z

    for i in range(21):  # 遍历该手的21个关键点

        # 获取3D坐标
        cx = int(hand_21.landmark[i].x * w)
        cy = int(hand_21.landmark[i].y * h)
        cz = hand_21.landmark[i].z
        depth_z = finesse_z - cz

        # 用圆的半径反映深度大小
        radius = max(int(6 * (1 + depth_z * 5)), 0)

        if i == 0:  # 手腕
            img = cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
        if i in [1, 5, 9, 13, 17]:  # 指根
            img = cv2.circle(img, (cx, cy), radius, (16, 144, 247), -1)
        if i in [2, 6, 10, 14, 18]:  # 第一指节
            img = cv2.circle(img, (cx, cy), radius, (1, 240, 255), -1)
        if i in [3, 7, 11, 15, 19]:  # 第二指节
            img = cv2.circle(img, (cx, cy), radius, (140, 47, 240), -1)
        if i in [4, 8, 12, 16, 20]:  # 指尖
            img = cv2.circle(img, (cx, cy), radius, (223, 155, 60), -1)



