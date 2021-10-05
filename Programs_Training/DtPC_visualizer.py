import open3d as o3d
from matplotlib import image
from matplotlib import pyplot
import numpy as np
"""CONSTANTS"""
path0 = 'C:/Users/maksn/Desktop/rgbd_dataset_freiburg1_xyz/'
Freiburg1 = [591.1, 590.1, 	331.0,	234.0, -0.0410, 0.3286,	0.0087,	0.0051, -0.5643]
cam_data = Freiburg1

fx, fy, cx, cy, d0, d1, d2, d3, d4 = cam_data
w = 480
h = 640

image_number = 84
"""USEFUL FUNCTIONS"""


def paint(picture):
    pyplot.imshow(picture)
    pyplot.show()


def show(variable):
    print(variable, sep="\n")


def quaternion_to_rotation_matrix(q):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    line_1 = [2*(q0**2+q1**2)-1, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)]
    line_2 = [2*(q1*q2+q0*q3), 2*(q0**2+q2**2)-1, 2*(q2*q3-q0*q1)]
    line_3 = [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 2*(q0**2+q3**2)-1]
    return np.matrix([line_1, line_2, line_3])


def matrix_ex(vector_array, rot_matrix):
    # show(vector_array)
    # show(rot_matrix)
    result = np.hstack([rot_matrix, vector_array])
    new_line = np.matrix([0 for i in range(rot_matrix.ndim+1)]+[1])
    result = np.vstack([result, new_line])
    # show(result)
    return np.linalg.inv(result)


def matrix_in(camera_params):
    fx = camera_params[0]
    fy = camera_params[1]
    x0 = camera_params[2]
    y0 = camera_params[3]
    s = 0
    result = np.array([[fx, s, x0],
                     [0., fy, y0],
                     [0., 0., 1.]])
    print(result)
    return result
# let's try to build point cloud using one image


"""STEP 1 - read files with image paths"""
with open(path0 + 'depth.txt', 'r') as depth_list:
    depth_images = depth_list.readlines()[3:]
    d_image = image.imread(path0 + 'depth/' + depth_images[image_number-1].rpartition(' ')[0] + '.png')
with open(path0 + 'groundtruth.txt', 'r') as gt:
    positions = gt.readlines()[3:]
    pose = positions[image_number-1].split()
    center_vector = np.matrix(list(map(float, pose[1:4]))).T
    rm = quaternion_to_rotation_matrix(list(map(float, pose[4:])))
    # print(center_vector)
    # print(positions[0])


"""STEP 2 - transform images to numpy arrays, construct Image objects of open3d from depth arrays - DONE"""
data_d = np.asarray(d_image)
image_depth = o3d.geometry.Image(data_d)

"""STEP 3 - generate camera intrinsic"""
intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
intrinsic.intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
cam = o3d.camera.PinholeCameraParameters()
cam.intrinsic = intrinsic
"""STEP 4 - generate camera extrinsic"""
cam.extrinsic = matrix_ex(center_vector, rm)
print(matrix_ex(center_vector, rm))
"""STEP 4 - create point cloud, using Image objects from step 3"""
pcd = o3d.geometry.PointCloud.create_from_depth_image(image_depth, cam.intrinsic, cam.extrinsic)
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

"""STEP 5 - set initial camera point at its real location and run visualizer"""
vis = o3d.visualization.Visualizer()
vis.create_window(height=intrinsic.height, width=intrinsic.width)
vis.add_geometry(pcd)

ctr = vis.get_view_control()
init_param = ctr.convert_to_pinhole_camera_parameters()
init_param.intrinsic = cam.intrinsic
init_param.extrinsic = cam.extrinsic
ctr.convert_from_pinhole_camera_parameters(init_param)
vis.run()
