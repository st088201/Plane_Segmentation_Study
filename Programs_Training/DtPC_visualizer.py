import open3d as o3d
from matplotlib import image
from matplotlib import pyplot
import numpy as np


def camera_object_from_tum_data(path0, camera_data, file_to_check, image_number):
    def quaternion_to_rotation_matrix(q):
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        line_1 = [2*(q0**2+q1**2)-1, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)]
        line_2 = [2*(q1*q2+q0*q3), 2*(q0**2+q2**2)-1, 2*(q2*q3-q0*q1)]
        line_3 = [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 2*(q0**2+q3**2)-1]
        return np.matrix([line_1, line_2, line_3])

    def matrix_extrinsic(vector_array, rot_matrix):
        # show(vector_array)
        # show(rot_matrix)
        result = np.hstack([rot_matrix, vector_array])
        new_line = np.matrix([0 for i in range(rot_matrix.ndim+1)]+[1])
        result = np.vstack([result, new_line])
        # show(result)
        return np.linalg.inv(result)

    with open(path0 + file_to_check, 'r') as gt:
        positions = gt.readlines()[3:]
        pose = positions[image_number-1].split()
        center_vector = np.matrix(list(map(float, pose[1:4]))).T
        rm = quaternion_to_rotation_matrix(list(map(float, pose[4:])))
    fx, fy, cx, cy, d0, d1, d2, d3, d4 = camera_data
    """generate camera intrinsic"""
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    intrinsic.intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = intrinsic
    """generate camera extrinsic"""
    cam.extrinsic = matrix_extrinsic(center_vector, rm)
    # print(matrix_extrinsic(center_vector, rm))
    return cam


def png_image_from_tum_data(path_to_data, img_type_str, img_number):
    """extract image with specific number from tum dataset; should choose depth or rgb"""
    with open(path_to_data + img_type_str+'.txt', 'r') as depth_list:
        depth_images = depth_list.readlines()[3:]
        d_image = image.imread(path_to_data + img_type_str+'/' + depth_images[img_number-1].rpartition(' ')[0] + '.png')
    return d_image


def point_cloud_from_o3d_depth_image(o3d_image_object_depth, o3d_camera_object):
    """create point cloud, using open3d.geometry.Image (pref. Depth) and open3d.camera objects"""
    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_image_object_depth,
                                                          o3d_camera_object.intrinsic,
                                                          o3d_camera_object.extrinsic,
                                                          depth_scale=5000.0,
                                                          depth_trunc=1000.0,
                                                          stride=1,
                                                          project_valid_depth_only=True)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd


def point_cloud_from_depth_image(img, cam_data):
    fx, fy, cx, cy, d0, d1, d2, d3, d4 = cam_data
    factor = 5000  # for the 16-bit PNG files
    points = np.zeros((640 * 480, 3))
    for u in range(1, 640, 1):
        for v in range(1, 480, 1):
            number = v * 640 + u
            points[number, 2] = img[v, u] / factor
            points[number, 0] = (u - cx) * points[number, 2] / fx
            points[number, 1] = (v - cy) * points[number, 2] / fy
    pcd_new = o3d.geometry.PointCloud()
    pcd_new.points = o3d.utility.Vector3dVector(points)
    return pcd_new


def visualizer_set_camera(vis_object, camera_object):
    ctr = vis_object.get_view_control()
    init_param = ctr.convert_to_pinhole_camera_parameters()
    init_param.intrinsic = camera_object.intrinsic
    init_param.extrinsic = camera_object.extrinsic
    ctr.convert_from_pinhole_camera_parameters(init_param)


if __name__ == "__main__":
    """Set constants"""
    dataset_path = 'C:/Users/maksn/Desktop/rgbd_dataset_freiburg1_xyz/'
    image_num = 84
    h = 480
    w = 640
    """Set the camera"""
    camera_parameters_Freiburg1 = [591.1, 590.1, 	331.0,	234.0, -0.0410, 0.3286,	0.0087,	0.0051, -0.5643]
    camera_data_file = 'groundtruth.txt'
    camera = camera_object_from_tum_data(dataset_path, camera_parameters_Freiburg1, camera_data_file, image_num)
    """Get the image, create clouds both ways"""
    image_depth_png = png_image_from_tum_data(dataset_path, 'depth', image_num)
    im_d = np.asarray(image_depth_png)
    cloud = point_cloud_from_o3d_depth_image(o3d.geometry.Image(im_d), camera)
    cloud_new = point_cloud_from_depth_image(im_d, camera_parameters_Freiburg1)
    """Visualizer"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Point cloud from depth image {}, created via open3d'.format(image_num),
                      width=camera.intrinsic.width,
                      height=camera.intrinsic.height)
    vis.add_geometry(cloud)
    # visualizer_set_camera(vis, camera)
    vis.run()

    vis_new = o3d.visualization.Visualizer()
    vis_new.create_window(window_name='Point cloud from depth image {}, created manually'.format(image_num),
                          height=camera.intrinsic.width,
                          width=camera.intrinsic.height)
    vis_new.add_geometry(cloud_new)
    # visualizer_set_camera(vis_new, camera)
    vis_new.run()
