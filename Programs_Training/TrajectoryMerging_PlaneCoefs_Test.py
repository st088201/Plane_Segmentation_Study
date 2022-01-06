import xml.etree.ElementTree as ET
import cv2
import itertools
import numpy as np
from random import randrange, uniform
from shapely.geometry import Polygon, Point
from matplotlib import image
import matplotlib.pyplot as plt
import open3d as o3d

"""FUNCTIONS BELOW USED FOR WORKING WITH XML MAPPING FILE AND EXTRACTING DATA FROM IT"""


def ann_points_to_2d_points_array(apt2dp_point_str):
    apt2dp_array_step_1 = apt2dp_point_str.split(";")
    apt2dp_array_step_2 = [[float(coor) for coor in apt2dp_array_step_1[i].split(",")] for i in
                           range(len(apt2dp_array_step_1))]
    return apt2dp_array_step_2


# returns list of [x, y] points transformed from certain polygon points part of xml file


def extract_all_polygons_border_points_from_frame_no(eap_ann_file, eap_frame_num):
    eap_ann_f_root = eap_ann_file.getroot()
    eap_planes_nums = []
    eap_planes_list = []
    for eap_plane in eap_ann_f_root.findall("./track/polygon/[@frame='{}']..".format(str(eap_frame_num))):
        eap_planes_nums.append(eap_plane.attrib["id"])
        for polygon in eap_plane.findall("./polygon/[@frame='{}']".format(str(eap_frame_num))):
            border_points = polygon.attrib["points"]
            eap_planes_list.append(ann_points_to_2d_points_array(border_points))
    return eap_planes_nums, eap_planes_list


# returns count of polygons and list of its borders like [[[x1,y1], [x2,y2], ...], ...] from n-th frame of ann file


def show_planes(sp_image_obj, sp_plane_array, sp_color_array, sp_title='Title'):
    color_count = 0
    for plane_points in sp_plane_array:
        plane_points_int32 = np.array([[int(round(num)) for num in point] for point in plane_points])
        cv2.fillPoly(sp_image_obj, pts=[plane_points_int32], color=sp_color_array[color_count % 100])
        color_count += 1
    cv2.imshow(sp_title, sp_image_obj)
    cv2.waitKey()


# imshow's all polygon forms from list of polygon borders on image object


"""FUNCTIONS BELOW USED FOR COMPARING PLANES OF TWO TRAJECTORY PARTS"""


def random_2d_inliers_list(ril_polygon_border_points, ril_inliers_number, ril_res_type='PointObj'):
    ril_poly = Polygon(ril_polygon_border_points)
    min_x, min_y, max_x, max_y = ril_poly.bounds
    ril_random_inliers = []
    while len(ril_random_inliers) < ril_inliers_number:
        random_point = Point([uniform(min_x, max_x), uniform(min_y, max_y)])
        if random_point.within(ril_poly):
            ril_random_inliers.append(random_point)
    if ril_res_type == 'coordinates':
        ril_random_inliers_coords = [[round(ril_point.x), round(ril_point.y)] for ril_point in ril_random_inliers]
        ril_random_inliers_coords.sort()
        list(ril_random_inliers_coords for ril_random_inliers_coords, _ in itertools.groupby(ril_random_inliers_coords))
        return ril_random_inliers_coords
    return ril_random_inliers


"""returns list of n random points lying inside specified border; can be PointObject or list of [x, y] pixels"""


def proportion_of_points_in_poly(pp_points_lst, pp_poly_borders):
    pp_poly = Polygon(pp_poly_borders)
    pp_inside_count = 0
    for pp_point in pp_points_lst:
        pp_inside_count += (pp_point.within(pp_poly))
    return pp_inside_count / len(pp_points_lst)


"""returns fraction of points of list which are located inside given polygon borders"""


def png_image_from_icl_data(png_im_num, image_type):
    with open(main_path + 'living_room_traj0_frei_png/associations.txt', 'r') as association_lst:
        association_line = association_lst.readlines()[png_im_num].split()
        pifid_image = image.imread(
            main_path + 'living_room_traj0_frei_png/' + association_line[1 * (image_type == 'depth') +
                                                                         3 * (image_type == 'rgb')]
        )
    return pifid_image


def camera_object_from_icl_data(path0, co_camera_data, file_to_check, image_number):
    def quaternion_to_rotation_matrix(q):
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        line_1 = [2 * (q0 ** 2 + q1 ** 2) - 1, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)]
        line_2 = [2 * (q1 * q2 + q0 * q3), 2 * (q0 ** 2 + q2 ** 2) - 1, 2 * (q2 * q3 - q0 * q1)]
        line_3 = [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 2 * (q0 ** 2 + q3 ** 2) - 1]
        return np.matrix([line_1, line_2, line_3])

    def matrix_extrinsic(vector_array, rot_matrix):
        result = np.hstack([rot_matrix, vector_array])
        new_line = np.matrix([0] * (rot_matrix.ndim + 1) + [1])
        result = np.vstack([result, new_line])
        return np.linalg.inv(result)

    with open(path0 + file_to_check, 'r') as gt:
        positions = gt.readlines()
        pose = positions[image_number - 1].split()
        center_vector = np.matrix(list(map(float, pose[1:4]))).T
        rm = quaternion_to_rotation_matrix(list(map(float, pose[4:])))
    fx, fy, cx, cy = co_camera_data
    """generate camera intrinsic"""
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    intrinsic.intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = intrinsic
    """generate camera extrinsic"""
    cam.extrinsic = matrix_extrinsic(center_vector, rm)
    return cam


def real_points_from_d_image_points_lst_rgbd_cam_data(rpd_depth_image, rpd_image_num, rpd_2d_points_lst,
                                                      rpd_groundtruth_frei_file,
                                                      rpd_cam_data=[481.20, -480.0, 319.50, 239.50]):
    def quaternion_to_rotation_matrix(q):
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        # print(q0, q1, q2, q3)
        rotation_matrix = [[2 * (q0 ** 2 + q1 ** 2) - 1, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
                           [2 * (q1 * q2 + q0 * q3), 2 * (q0 ** 2 + q2 ** 2) - 1, 2 * (q2 * q3 - q0 * q1)],
                           [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 2 * (q0 ** 2 + q3 ** 2) - 1]]
        # print(np.array(rotation_matrix))
        return np.array(rotation_matrix)

    def matrix_extrinsic(vector_array, rotation_matrix):
        result = np.hstack([rotation_matrix, vector_array])
        new_line = np.matrix([0] * (rotation_matrix.ndim + 1) + [1])
        result = np.vstack([result, new_line])
        return np.linalg.inv(result)

    with open(main_path + 'living_room_traj0_frei_png/' + rpd_groundtruth_frei_file, 'r') as gt:
        positions = gt.readlines()[:]
        # print(positions[0])
        pose = positions[rpd_image_num].split()
        # print('We are checking image num {}, associated pose is {}'.format(rpd_image_num, pose))
        center_vector = np.matrix(list(map(float, pose[1:4]))).T
        rot_matrix = quaternion_to_rotation_matrix(list(map(float, pose[4:])))
    fx, fy, cx, cy = rpd_cam_data
    """generate camera intrinsic"""
    matrix_intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # print(matrix_intr)
    """generate camera extrinsic, this part is mostly useless for now 
    as center_vector and rot_matrix are used later separately"""
    matrix_extr = matrix_extrinsic(center_vector, rot_matrix)[:3]
    # print(matrix_extr)
    # camera_matrix_extended = np.vstack((matrix_intr * matrix_extr, np.array([0, 0, 0, 1])))
    # print(camera_matrix_extended)
    # print('Inversed changed camera matrix\n', np.linalg.inv(camera_matrix_extended))
    rpd_real_points_lst = np.zeros((640 * 480, 3))
    factor = 1
    count = 0
    for rpd_2d_point in rpd_2d_points_lst:
        x2, y2 = rpd_2d_point
        x2 = x2 * (x2 < 640) + 639 * (x2 >= 640)
        y2 = y2 * (y2 < 480) + 479 * (y2 >= 480)
        # print("Point coordinates {}, {}".format(x2, y2))
        rpd_depth = rpd_depth_image[y2, x2]
        imagined_point = np.array(
            [x2 * rpd_depth / factor, y2 * rpd_depth / factor, rpd_depth / factor]
        )
        inter_point = np.dot(np.linalg.inv(matrix_intr), imagined_point.T)
        rpd_real_point = np.dot(rot_matrix, inter_point-center_vector)
        # print('Real point is\n', rpd_real_point)
        for i in range(3):
            rpd_real_points_lst[count, i] = rpd_real_point[0, i]
        count += 1
    return rpd_real_points_lst, np.array([0, 0, 0, 1])


"""this function is possibly invalid, should be reworked or deleted later"""


def real_points_from_d_image_points_lst_icl_cam_data(rpd_depth_image, rpd_image_num, rpd_2d_points_lst,
                                                     rpd_extr_matrix_data_file,
                                                     rpd_cam_data=[481.20, -480.0, 319.50, 239.50]):
    with open(main_path + 'living_room_traj0_frei_png/' + rpd_extr_matrix_data_file, 'r') as gt:
        positions = gt.readlines()
        extr_matrices = [positions[i:i+3] for i in range(0, len(positions), 4)]
        matrix_extr = np.array([pos_line.split() for pos_line in extr_matrices[rpd_image_num-1]], dtype=float)
        """Not sure about number of matrices, therefore -1 in previous line"""
        # print('image {}, extr matrix\n{}'.format(rpd_image_num, matrix_extr))
    fx, fy, cx, cy = rpd_cam_data
    matrix_intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    print(matrix_extr)
    rpd_real_points_lst = np.zeros((640 * 480, 3))
    count = 0
    for rpd_2d_point in rpd_2d_points_lst:
        x2, y2 = rpd_2d_point
        x2 = x2 * (x2 < 640) + 639 * (x2 >= 640)
        y2 = y2 * (y2 < 480) + 479 * (y2 >= 480)
        # print("Point coordinates ({}, {})".format(x2, y2))
        z = rpd_depth_image[y2, x2]
        imagined_point = np.array([x2 * z, y2 * z, z])
        r1 = np.dot(np.linalg.inv(matrix_intr), imagined_point.T)
        # print('r1 is', r1)
        """Until here coordinate transformation is correct; r1 is vector in CS tied to camera"""
        rpd_real_point = np.dot(matrix_extr[:, :3], r1) + matrix_extr[:, 3].T
        """What is wrong with camera coordinates in world frame?"""
        # print('Real point is\n', rpd_real_point)
        for i in range(3):
            rpd_real_points_lst[count, i] = rpd_real_point[i]
        count += 1
    print('Last real point is', rpd_real_point)
    return rpd_real_points_lst, matrix_extr[:, 3]


def plane_coefs_from_3_point_array(pc_point_array):
    p1, p2, p3 = pc_point_array
    v1 = p3 - p1
    v2 = p2 - p1
    cp = np.cross(v1, v2)
    a, b, c = cp
    d = np.dot(cp, p3)
    print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))


"""Above is vastly less accurate, more fast version of next function.
After RT problem solving will be tested whether it can give me required accuracy"""


def plane_coefs_from_point_array(pc_point_array, pc_method):
    inliers_3d_pcd = o3d.geometry.PointCloud()
    inliers_3d_pcd.points = o3d.utility.Vector3dVector(pc_point_array)
    plane_model, inliers = inliers_3d_pcd.segment_plane(distance_threshold=10, ransac_n=3, num_iterations=1000)
    return plane_model


def plt_show_image(ci_image_obj):
    imgplot = plt.imshow(ci_image_obj)
    plt.show()


def build_pcd_from_image(part_number, image_number_in_part):
    b_im_num = 299 * (part_number == 1) + 999 * (part_number == 2) + image_number_in_part
    b_image = png_image_from_icl_data(b_im_num, 'depth')
    all_pixels_lst = [[4 * int(v / 120), 4 * (v % 120)] for v in range(160 * 120)]
    """to speed the process we use only part of image pixels, therefore 640/4=160, 480/4=120"""
    test_pcd_points, misc_vect = real_points_from_d_image_points_lst_icl_cam_data(np.asarray(b_image), b_im_num, all_pixels_lst, 'livingRoom0.gt.extrdata.txt')
    # test_pcd_points, misc_vect = real_points_from_d_image_points_lst_rgbd_cam_data(np.asarray(b_image), b_im_num, all_pixels_lst, 'livingRoom0.gt.freiburg')
    """misc vector from icl part is for testing, from rgbd is placeholder for now"""
    test_pcd = o3d.geometry.PointCloud()
    test_pcd.points = o3d.utility.Vector3dVector(test_pcd_points)
    return test_pcd, misc_vect


def list_of_plane_coefs(ann_file, inliers_number, occuring_frame_num=0):
    lpc_root = ann_file.getroot()
    lpc_plane_coefs_lst = []
    lpc_file_name = lpc_root.findall("./meta/task")[0].find('name').text
    lpc_lowest_file_num = int(lpc_file_name.split()[1].split('-')[0])
    # print('Calculating plane coefficients for all planes in annotation file', lpc_file_name)
    for lpc_plane in lpc_root.findall('./track'):
        # print('plane id', lpc_plane.attrib["id"])
        if len(lpc_plane) >= occuring_frame_num:
            lpc_polygon = lpc_plane[occuring_frame_num]
        else:
            lpc_polygon = lpc_plane[0]
        lpc_image_num = int(lpc_polygon.attrib["frame"]) + lpc_lowest_file_num
        lpc_acc_img = png_image_from_icl_data(lpc_image_num, 'depth')
        lpc_border = ann_points_to_2d_points_array(lpc_polygon.attrib["points"])
        lpc_inliers_2d = random_2d_inliers_list(lpc_border,
                                                ril_inliers_number=inliers_number,
                                                ril_res_type='coordinates')
        lpc_inliers_3d = real_points_from_d_image_points_lst_icl_cam_data(np.asarray(lpc_acc_img),
                                                                          lpc_image_num,
                                                                          lpc_inliers_2d,
                                                                          'livingRoom0.gt.extrdata.txt')
        lpc_plane_coefs_RANSAC = plane_coefs_from_point_array(lpc_inliers_3d, pc_method='RANSAC')
        lpc_plane_coefs_leastsquares = plane_coefs_from_point_array(lpc_inliers_3d, pc_method='leastsquares')
        lpc_last_coef = lpc_plane_coefs_leastsquares[3]
        lpc_result_coefs = [coef / lpc_last_coef for coef in lpc_plane_coefs_leastsquares]
        lpc_plane_coefs_lst.append(lpc_result_coefs)
        # print('Plane {}, coefs {}'.format(lpc_plane.attrib["id"], lpc_plane_coefs))
    return lpc_plane_coefs_lst


if __name__ == '__main__':
    main_path = "C:/Users/maksn/Desktop/PE files/Living Room Dataset/traj0/"
    ann_file_1 = ET.parse(main_path + "Annotations/0-299/annotations_0_reworked.xml")
    ann_file_2 = ET.parse(main_path + "Annotations/300-999/annotations_1_reworked.xml")
    ann_file_3 = ET.parse(main_path + "Annotations/1000-1509/annotations_2_reworked.xml")
    w = 480
    h = 640
    image_blank = np.zeros(shape=(480, 640, 3), dtype=np.uint8)
    colors_list = [tuple([randrange(255), randrange(255), randrange(255)]) for i in range(100)]
    camera_data = [481.20, -480.0, 319.50, 239.50]
    cloud1, aa = build_pcd_from_image(part_number=0, image_number_in_part=1)
    cloud2, bb = build_pcd_from_image(part_number=0, image_number_in_part=20)
    print('Difference is\n', (np.asarray(cloud2.points)-np.asarray(cloud1.points))[:6])
    print('T difference is', aa-bb)
    o3d.visualization.draw_geometries([cloud1, cloud2], window_name='Point cloud from according depth image')
