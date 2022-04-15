import open3d

################## public ##########################

axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame()
axis_pcd.scale(0.1, center=axis_pcd.get_center())
pcd: open3d.geometry.PointCloud = open3d.geometry.PointCloud()  # 定义点云
################## blocking ###########################


def show_3d_blocking(xyz):
    # 方法2（阻塞显示）：调用draw_geometries直接把需要显示点云数据
    pcd.points = open3d.utility.Vector3dVector(xyz)  # 定义点云坐标位置
    # open3d.visualization.draw_geometries([pcd] + [axis_pcd], window_name="blocking")
    open3d.visualization.draw_geometries([pcd], window_name="blocking",width=600,height=600,mesh_show_wireframe=True)


################## non-blocking ###########################

vis = open3d.visualization.Visualizer()
vis.create_window(width=600, height=600,window_name="non_blocking")
vis.add_geometry(axis_pcd)

# control = vis.get_view_control()
# control.scale(1000)

render_option: open3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
render_option.point_size = 2.0  # 设置渲染点的大小
# render_option.show_coordinate_frame = True


# 非阻塞式
def show_3d(xyz):
    # point3D二维numpy矩阵,将其转换为open3d点云格式
    pcd.points = open3d.utility.Vector3dVector(xyz)

    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
