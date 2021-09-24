def depth2xyz(u, v, depthValue ,fx,fy,cx,cy):
    fx = 361.1027
    fy = 361.8266
    cx = 258.4545
    cy = 212.1282

    depth = depthValue * 0.001  # depth2mi

    z = float(depth)
    x = float((u - cx) * z) / fx
    y = float((v - cy) * z) / fy

    result = [x, y, z]
    return result