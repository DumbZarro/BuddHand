import numpy as np

from .YoloxConfig import yolox


def detect_hand(color_image):
    outputs, img_info = yolox.inference(color_image)
    hand_results = []
    if outputs == [None]:
        return hand_results
    # print(outputs)
    # print(outputs[0])

    bbox_list = outputs[0].cpu().numpy().tolist()
    for item in bbox_list:
        hand_results.append(dict(bbox=np.array(item[:5], dtype=np.float32)))
    # print("hand_results:")
    # print(hand_results)

    return hand_results
