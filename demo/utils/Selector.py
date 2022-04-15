from .HandDetector import detect_hand


def select_stream(flag, color_image, MBR):
    if flag:  # 上一帧有检测到手势,使用上一帧的最小外接矩形
        return True, MBR  # 同时默认这一帧也能检测到,检测不到后续再给False
    else:
        # detect hand
        hand_results = detect_hand(color_image)
        if hand_results == []:
            return False, []
        else:
            return True, hand_results  # 检测到手势
