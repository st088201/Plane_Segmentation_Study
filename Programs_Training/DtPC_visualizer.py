import open3d as o3d
from matplotlib import image
from math import trunc
import numpy as np
from os import path
from random import random


def camera_object_from_tum_data(path0, camera_data, file_to_check, image_number):
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
        positions = gt.readlines()[3:]
        pose = positions[image_number - 1].split()
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
    return cam


def png_image_from_tum_data(path_to_data, img_type_str, img_number):
    """extract image with specific number from tum dataset; should choose depth or rgb"""
    with open(path_to_data + img_type_str + '.txt', 'r') as depth_list:
        depth_images = depth_list.readlines()[3:]
        d_image = image.imread(
            path.join(path_to_data + img_type_str, depth_images[img_number - 1].rpartition(' ')[0] + '.png'))
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
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
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


"""FUNCTIONS BELOW UNDER CONSTRUCTION"""


def point_arr_projected(point_array, plane_model_measures):
    # print('PROJECTING: point_array\n', point_array)
    # project dot onto plane for future use: evade clusters growing perpendicular to plane
    projected_points = np.zeros((640 * 480, 3))
    p_a, p_b, p_c, p_d = plane_model_measures.ravel()
    for i in range(len(point_array)):
        x0, y0, z0 = point_array[i][0], point_array[i][1], point_array[i][2]
        x = (p_a * p_d - p_a * (p_b * y0 + p_c * z0) + (p_b ** 2 + p_c ** 2) * x0) / (p_a ** 2 + p_b ** 2 + p_c ** 2)
        y = y0 + (p_b / p_a) * (x - x0)
        z = z0 + (p_c / p_a) * (x - x0)
        projected_points[i][0] = x
        projected_points[i][1] = y
        projected_points[i][2] = z
    # print('PROJECTING: projected_points\n', projected_points)
    return projected_points


def change_co_sys_3d_to_2d(point_array, plane_model_measures):
    # generating vectors are of different length, fucker!
    projected_points = np.zeros((len(point_array), 2))
    p_a, p_b, p_c, p_d = plane_model_measures.ravel()
    for i in range(len(point_array)):
        y0, z0 = point_array[i][1], point_array[i][2]
        projected_points[i][0] = (y0 + p_c * z0 / p_b + p_d / p_b) * (-(p_a ** 2 + p_b ** 2) ** 0.5) / p_a
        projected_points[i][1] = (z0 + p_d / (2 * p_c)) * (
            -((p_a * p_b) ** 2 + (p_c * p_b) ** 2 + (p_a * p_c) ** 2) ** 0.5) / (p_a * p_b)
    return projected_points


def find_borders(plane_points_array):
    # print('BORDERS: plane_points_array:\n', plane_points_array)
    x_smallest, x_biggest = plane_points_array[0][0], plane_points_array[0][0]
    y_smallest, y_biggest = plane_points_array[0][1], plane_points_array[0][1]
    for i in range(1, len(plane_points_array)):
        x_smallest = min(x_smallest, plane_points_array[i][0])
        x_biggest = max(x_biggest, plane_points_array[i][0])
        y_smallest = min(y_smallest, plane_points_array[i][1])
        y_biggest = max(y_biggest, plane_points_array[i][1])
    return x_smallest, x_biggest, y_smallest, y_biggest


def euclid_distance_n_dims(array_n_dims, num1, num2):
    return (sum([(array_n_dims[num1][i] - array_n_dims[num2][i]) ** 2 for i in range(len(array_n_dims[0]))])) ** 0.5


def most_common_of_list(lst):
    my_dict = {}
    cnt, itm = 0, ''
    for item in reversed(lst):
        my_dict[item] = my_dict.get(item, 0) + 1
        if my_dict[item] >= cnt:
            cnt, itm = my_dict[item], item
    return itm


def compare_cells(cc_max_dist, cc_proj_point_array, cc_dist_func,
                  cc_cell_1_num, cc_cell_2_num,
                  cc_cell_1_points, cc_cell_2_points, cc_merged_cells_dict):
    print("Comparing cells {} and {}, {} and {} points respectively".format(cc_cell_1_num,
                                                                            cc_cell_2_num,
                                                                            len(cc_cell_1_points),
                                                                            len(cc_cell_2_points)))
    for point_1 in cc_cell_1_points:
        for point_2 in cc_cell_2_points:
            if cc_dist_func(cc_proj_point_array, point_1, point_2) < cc_max_dist:
                cc_merged_cells_dict[cc_cell_1_num].append(cc_cell_2_num)
                print('Cells to be merged: {}:{}'.format(cc_cell_1_num, cc_merged_cells_dict[cc_cell_1_num]))
                break
            if cc_dist_func(cc_proj_point_array, point_1, point_2) > 2 * cc_max_dist:
                break
        else:
            continue
        break
    """Function to compare two cells and decide if they need to be merged; 
    includes fool-proof condition to stop checking if cells are definitely not adjacent"""


def list_of_point_lists_by_affinity(lpl_aff_list, lpl_filled_cell_nums):
    lpl_max_aff_num = lpl_filled_cell_nums[-1]
    lpl_list_of_point_lists = [[] for lpl_step in range(len(lpl_filled_cell_nums))]
    lpl_cell_num_to_filled_cell_order = {lpl_filled_cell_nums[lpl_step]: lpl_step
                                         for lpl_step in range(len(lpl_filled_cell_nums))}
    # print(lpl_cell_num_to_filled_cell_order)
    print("GROUPING LISTS: NUMBER OF AFFINITIES {}; "
          "MAXIMUM AFFINITY OUTSIDE {}, INSIDE {}, ALSO {}".format(len(sorted(list(dict.fromkeys(lpl_aff_list)))),
                                                                   lpl_max_aff_num, max(lpl_aff_list),
                                                                   max(lpl_filled_cell_nums)))
    for lpl_point_num in range(len(lpl_aff_list)):
        lpl_cell_number_in_filled_cells = lpl_cell_num_to_filled_cell_order[lpl_aff_list[lpl_point_num]]
        lpl_list_of_point_lists[lpl_cell_number_in_filled_cells].append(lpl_point_num)
        # print("Point number {}, affinity {}, total {} points in the same cell".format(lpl_point_num, lpl_aff_list[lpl_point_num], len(lpl_list_of_point_lists[lpl_cell_number_in_filled_cells])))
    for lpl_filled_cell in lpl_filled_cell_nums:
        print("Cell no. {}, count {}, total points {}".format(lpl_filled_cell,
                                                              lpl_cell_num_to_filled_cell_order[lpl_filled_cell], len(
                lpl_list_of_point_lists[lpl_cell_num_to_filled_cell_order[lpl_filled_cell]])))
    return lpl_list_of_point_lists


def inverted_dictionary(id_dict):
    id_inverted_dict = {}
    id_dict_keys_list = []
    id_dict_keys_list[:] = list(id_dict.keys())[:]
    for id_key in id_dict_keys_list:
        for id_key_value in id_dict[id_key]:
            id_inverted_dict[id_key_value] = id_key
        id_inverted_dict[id_key] = id_key
    return id_inverted_dict


def rebuild_aff_list_with_connectivity_dict(ral_list, ral_dict):
    ral_dict_final = inverted_dictionary(find_classes_in_dict(ral_dict))
    print(ral_dict_final)
    for ral_i in range(len(ral_list)):
        ral_list[ral_i] = ral_dict_final[ral_list[ral_i]]


def find_classes_in_dict(fc_dict):
    def snap(s_dict):
        for s_key in s_dict.keys():
            s_dict[s_key] = list(set(s_dict[s_key]))
    """Function to remove duplicates from key value lists"""

    def find_all_children(f_dict, f_parent, f_children):
        if not (f_parent in f_children):
            f_children.append(f_parent)
            for f_child in f_dict[f_parent]:
                find_all_children(f_dict, f_child, f_children)
        else:
            return
    """Function to extract all descendants of a key into a list"""

    fc_im_dict = fc_dict.copy()
    for key in fc_dict.keys():
        for value in fc_dict[key]:
            if fc_dict.get(value):
                fc_im_dict[value].append(key)
            else:
                fc_im_dict[value] = [key]
    fc_dict = fc_im_dict.copy()
    snap(fc_dict)
    """Above we made our graph non-oriented and removed duplicates from its key values"""
    """Creating new dict of classes"""
    c_dict = {}
    keys_to_consider = [key for key in fc_dict.keys()]
    while keys_to_consider:
        key = keys_to_consider[0]
        key_area = []
        find_all_children(fc_dict, key, key_area)
        c_dict[key] = key_area
        keys_to_consider.remove(key)
        for value in key_area:
            if value in keys_to_consider:
                keys_to_consider.remove(value)
    return c_dict


def biggest_plane_point_group_nums(proj_point_array, chosen_points_nums_list, distance_function):
    chosen_points_count = len(chosen_points_nums_list)
    affinity_list = [0] * chosen_points_count

    def show_affinity(proj_point_arr, chosen_dot_num_list, affinity_array, title):
        aff_group_count = max(affinity_array)
        aff_group_list = [o3d.geometry.PointCloud for sa_step in range(aff_group_count)]
        for sa_count in range(aff_group_count):
            group_cloud = o3d.geometry.PointCloud()
            group_cloud.points = o3d.utility.Vector3dVector([proj_point_arr[k] for k in
                                                             [dot_num for dot_num in range(len(chosen_dot_num_list))
                                                              if affinity_array[dot_num] == sa_count]])
            group_cloud.paint_uniform_color([random(), random(), random()])
            aff_group_list[sa_count] = group_cloud

        o3d.visualization.draw_geometries(aff_group_list, window_name=title)

    print('GROUPING: chosen_points_num_list size:{}, proj_point_array_size:{}'.format(chosen_points_count,
                                                                                      proj_point_array.shape[0]))
    """STEP 1: divide plane into cells"""
    print('STEP 1: dividing plane into cells')
    x_min, x_max, y_min, y_max = find_borders(proj_point_array)
    max_dist_allowed = 0.007 * ((x_max - x_min) ** 2 + (y_max - y_min) ** 2) ** 0.5
    print("Max_dist_allowed is", max_dist_allowed)
    print('STEP 1: borders are ({0};{1}) and ({2};{3}), with respect to X and Y'.format(x_min, x_max, y_min, y_max))
    step_count_x = round((x_max - x_min) / max_dist_allowed) + 1
    step_count_y = round((y_max - y_min) / max_dist_allowed) + 1
    print('STEP 1: step counts by x and by y are', step_count_x, step_count_y)

    """STEP 2: assign points to newfound cells"""
    print('STEP 2: assigning points to newfound cells')
    # print("Cell assigning for points number", end=' ')
    for i in range(chosen_points_count):
        x0 = proj_point_array[i][0]
        y0 = proj_point_array[i][1]
        cell_num_by_x = trunc((x0 - x_min) * step_count_x / (x_max - x_min)) - 1 * (x0 == x_max)
        cell_num_by_y = trunc((y0 - y_min) * step_count_y / (y_max - y_min)) - 1 * (y0 == y_max)
        # print("For point number {} coordinates are {} and {}".format(i, cell_num_by_x, cell_num_by_y))
        affinity_list[i] = cell_num_by_x + cell_num_by_y * step_count_x
        """Our cell affinity goes like this, only with respect to step counts:
            [9,10,11]
            [6, 7, 8]
            [3, 4, 5]
            [0, 1, 2]"""
    print('STEP 2: affinity_list_by_cells, first couple of points', affinity_list[:10])
    # show_affinity(proj_point_array, chosen_points_nums_list, affinity_list, title='Affinity after step 2 - divided by cells')

    """STEP 3: check adjacent cells for every cell and merge cells with points close enough to each other"""
    print('STEP 3: checking adjacent cells for every cell and merging points close enough to each other')
    filled_cell_nums = [int(i) for i in sorted(list(dict.fromkeys(affinity_list)))]
    print('filled cell nums type is', type(filled_cell_nums[0]))
    list_of_points_by_aff = list_of_point_lists_by_affinity(affinity_list, filled_cell_nums)
    cells_to_merge_dict = {dict_gen_cell_num: [] for dict_gen_cell_num in filled_cell_nums}
    cell_num_to_filled_cell_order = {filled_cell_nums[step]: step for step in range(len(filled_cell_nums))}
    print('STEP 3: non-empty cells', filled_cell_nums)
    print('STEP 3: list of point lists by affinity: last cell contains', len(list_of_points_by_aff[-1]))
    for i in range(step_count_y):
        for j in range(step_count_x):
            main_cell_num = i * step_count_x + j
            if main_cell_num in filled_cell_nums:
                print('Checking non-empty cell no.', main_cell_num)
                adj_cell_nums = [main_cell_num - 1,
                                 main_cell_num + step_count_x - 1,
                                 main_cell_num + step_count_x,
                                 main_cell_num + step_count_x + 1,
                                 main_cell_num + 1]
                for adj_cell_num in adj_cell_nums:
                    if adj_cell_num >= 0 and adj_cell_num in filled_cell_nums:
                        print("Processing points in adjacent cell number", adj_cell_num)
                        compare_cells(max_dist_allowed, proj_point_array, distance_function,
                                      main_cell_num, adj_cell_num,
                                      list_of_points_by_aff[cell_num_to_filled_cell_order[main_cell_num]],
                                      list_of_points_by_aff[cell_num_to_filled_cell_order[adj_cell_num]],
                                      cells_to_merge_dict)
    print("STEP 3: NUMBERS OF CELLS THAT CAN BE MERGED", cells_to_merge_dict)

    """STEP 4: TRACING ALL CELLS-TO-BE-MERGED USING BUILT DICTIONARY"""
    rebuild_aff_list_with_connectivity_dict(affinity_list, cells_to_merge_dict)
    # show_affinity(proj_point_array, chosen_points_nums_list, affinity_list, title='Affinity after step 4 - classes found')

    """STEP 5: finding biggest affinity group"""
    """so now we got affinity list by groups, something like [2, 17, 2, 25, 34, 7, 7, ...];
    Our next goal is to find biggest cluster"""
    print('\nSTEP 5: finding biggest cluster')
    print('STEP 5: affinity_list_by_groups, 20 first points', affinity_list[:20])
    most_freq_affinity = most_common_of_list(affinity_list)
    print('STEP 5: most frequent affinity', most_freq_affinity)
    biggest_cluster_nums = []
    most_freq_affinity_count = 0
    for i in range(chosen_points_count):
        if affinity_list[i] == most_freq_affinity:
            biggest_cluster_nums.append(chosen_points_nums_list[i])
            most_freq_affinity_count += 1
    print('RESULT: OF TOTAL {} POINTS {} LIE IN BIGGEST FOUND CLUSTER'.format(len(affinity_list),
                                                                              most_freq_affinity_count))
    return biggest_cluster_nums


if __name__ == "__main__":
    """Set constants"""
    dataset_path = 'C:/Users/maksn/Desktop/rgbd_dataset_freiburg1_xyz/'
    image_num = 82
    h = 480
    w = 640
    """Set the camera"""
    camera_parameters_Freiburg1 = [591.1, 590.1, 331.0, 234.0, -0.0410, 0.3286, 0.0087, 0.0051, -0.5643]
    camera_data_file = 'groundtruth.txt'
    camera = camera_object_from_tum_data(dataset_path, camera_parameters_Freiburg1, camera_data_file, image_num)
    """Get the image, create clouds both ways"""
    image_depth_png = png_image_from_tum_data(dataset_path, 'depth', image_num)
    im_d = np.asarray(image_depth_png)
    cloud = point_cloud_from_o3d_depth_image(o3d.geometry.Image(im_d), camera)
    # cloud_new = point_cloud_from_depth_image(im_d, camera_parameters_Freiburg1)
    # o3d.visualization.draw_geometries([cloud])
    """Plane extraction on point cloud above"""
    planes_extracted = []
    plane_model, inliers = cloud.segment_plane(distance_threshold=0.003, ransac_n=3, num_iterations=1000)
    outlier_cloud = cloud
    unaffected_points_count = w * h
    while unaffected_points_count > (w * h) * 0.3:
        """condition to stop searching for point clouds, could and should be changed"""
        [a, b, c, d] = plane_model
        print("Plane equation: {:.5f}x + {:.5f}y + {:.5f}z + {:.5f} = 0".format(a, b, c, d))
        inlier_cloud = outlier_cloud.select_by_index(inliers)
        o3d.visualization.draw_geometries([inlier_cloud], window_name='Points close enough to plane')

        """next we project found points onto plane, then
        change coordinate system for more simple subsequent calculations"""
        inlier_cloud_points_projected = point_arr_projected(np.asarray(inlier_cloud.points), plane_model)
        projected_points_cloud = o3d.geometry.PointCloud()
        projected_points_cloud.points = o3d.utility.Vector3dVector(inlier_cloud_points_projected)
        # o3d.visualization.draw_geometries([projected_points_cloud], window_name='Close points projected onto plane in 3d')

        inlier_cloud_points_projected_2d = change_co_sys_3d_to_2d(inlier_cloud_points_projected, plane_model)
        added_line = np.array([[0]] * inlier_cloud_points_projected_2d.shape[0])
        np.transpose(added_line)
        inlier_cloud_points_projected_in3d = np.hstack([inlier_cloud_points_projected_2d,
                                                        added_line])
        projected_points_cloud = o3d.geometry.PointCloud()
        projected_points_cloud.points = o3d.utility.Vector3dVector(inlier_cloud_points_projected_in3d)
        # o3d.visualization.draw_geometries([projected_points_cloud],window_name='Coordinate system changed to 2d, z==0 for every point')

        plane_found_nums = biggest_plane_point_group_nums(inlier_cloud_points_projected,
                                                          inliers,
                                                          euclid_distance_n_dims)
        """we found biggest form of points in plane, paint it random, 
        add plane to list of planes we got and repeat the process with left points"""
        unaffected_points_count -= len(plane_found_nums)
        plane_found = outlier_cloud.select_by_index(plane_found_nums)
        plane_found.paint_uniform_color([random(), random(), random()])
        # o3d.visualization.draw_geometries([plane_found], window_name='Form found')

        planes_extracted.append(plane_found)
        outlier_cloud = outlier_cloud.select_by_index(plane_found_nums, invert=True)
        outlier_cloud.paint_uniform_color([random(), random(), random()])
        o3d.visualization.draw_geometries([outlier_cloud, plane_found], window_name='Point cloud with highlighted extracted plane')

        plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.003, ransac_n=3, num_iterations=1000)
    o3d.visualization.draw_geometries(planes_extracted + [outlier_cloud], window_name='Result')
