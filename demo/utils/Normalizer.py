from .CameraConfig import w,h

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
