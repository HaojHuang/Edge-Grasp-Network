import numpy as np
import torch
import open3d as o3d
from vis_grasp import draw_scene
from mayavi import mlab
import torch.nn.functional as F

def downsample_points(pts, K):
    # if num_pts > 2K use farthest sampling
    # else use random sampling
    if pts.shape[0] >= 2*K:
        sampler = FarthestSampler()
        return sampler(pts, K)
    else:
        return pts[np.random.choice(pts.shape[0], K, replace=(K<pts.shape[0])), :]

class FarthestSampler:
    def __init__(self):
        pass

    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def __call__(self, pts, k):
        index_list = []
        farthest_pts = np.zeros((k, 3), dtype=np.float32)
        index = np.random.randint(len(pts))
        farthest_pts[0] = pts[index]
        index_list.append(index)
        distances = self._calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            index = np.argmax(distances)
            farthest_pts[i] = pts[index]
            index_list.append(index)
            distances = np.minimum(
                distances, self._calc_distances(farthest_pts[i], pts))
        return farthest_pts, index_list


def get_geometry_mask(normals_dot,dot_product_2,relative_pos,des_normals,sample_normal,sample_pos,pos,use_o3d=False,strict=False):
    '''
    function to calculate neighbors that satisfy the four constrains:
    1. the radius constrain is already satisfied
    :param normals_dot:
    :param dot_product_2:
    :param relative_pos:
    :param des_normals:
    :param sample_normal:
    :param sample_pos:
    :param pos: (N,3) of the pcd
    :param use_o3d: bool (if true use open3d to calculate the distances matrix between two set of points, otherwise use torch.cdist)
    :param strict:
    :return:
    '''

    orth_mask = abs(normals_dot) < 0.1
    pitch_constrain_mask = abs(dot_product_2) < 0.1
    orth_mask = orth_mask.squeeze(dim=-1)
    pitch_constrain_mask = pitch_constrain_mask.squeeze(dim=-1)
    angle_mask = torch.logical_and(orth_mask,pitch_constrain_mask)
    half_baseline_projection = torch.sum(relative_pos * des_normals, dim=-1)
    depth_projection = -torch.sum(relative_pos * sample_normal, dim=-1)
    geometry_mask_1 = torch.logical_and(0.005 < half_baseline_projection, half_baseline_projection < 0.038)
    geometry_mask_2 = torch.logical_and(0.005 < depth_projection, depth_projection < 0.043)
    geometry_mask = torch.logical_and(geometry_mask_1, geometry_mask_2)
    geometry_mask = torch.logical_and(geometry_mask, angle_mask)
    side_points_1 = -des_normals * 0.04 + sample_pos
    side_points_2 = des_normals * 0.04 + sample_pos
    center_dis_from_source = (0.105 - 0.059- depth_projection).unsqueeze(dim=-1)
    if strict:
        # more strict collision check
        gripper_center = center_dis_from_source.repeat(1, 3) * sample_normal + sample_pos
        side_points_1_end = -des_normals * 0.04 + (gripper_center - 0.04627*sample_normal)
        side_points_2_end =  des_normals * 0.04  +  (gripper_center -0.04627 * sample_normal)
    if use_o3d:
        # use open3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pos.numpy())
        #pcd.normals = o3d.utility.Vector3dVector(batch.normals.numpy())
        # o3d.visualization.draw_geometries([pcd])
        side_pcd_1 = o3d.geometry.PointCloud()
        side_pcd_1.points = o3d.utility.Vector3dVector(side_points_1.numpy())
        side_pcd_2 = o3d.geometry.PointCloud()
        side_pcd_2.points = o3d.utility.Vector3dVector(side_points_2.numpy())
        dists_1 = side_pcd_1.compute_point_cloud_distance(pcd)
        dists_1 = torch.from_numpy(np.asarray(dists_1))
        dists_2 = side_pcd_2.compute_point_cloud_distance(pcd)
        dists_2 = torch.from_numpy(np.asarray(dists_2))
        no_collision_mask_o3d = torch.logical_and(dists_1 > 0.01, dists_2 > 0.01)
        if strict:
            #more strict collision check
            gripper_center_pcd = o3d.geometry.PointCloud()
            gripper_center_pcd.points = o3d.utility.Vector3dVector(gripper_center.numpy())
            center_dist = gripper_center_pcd.compute_point_cloud_distance(pcd)
            side_pcd_1_end = o3d.geometry.PointCloud()
            side_pcd_1_end.points = o3d.utility.Vector3dVector(side_points_1_end.numpy())
            side_pcd_2_end = o3d.geometry.PointCloud()
            side_pcd_2_end.points = o3d.utility.Vector3dVector(side_points_2_end.numpy())
            dists_1_end = side_pcd_1_end.compute_point_cloud_distance(pcd)
            dists_2_end = side_pcd_2_end.compute_point_cloud_distance(pcd)
            center_dist = torch.from_numpy(np.asarray(center_dist))
            dists_1_end = torch.from_numpy(np.asarray(dists_1_end))
            dists_2_end = torch.from_numpy(np.asarray(dists_2_end))
            dist_end_mask = torch.logical_and(dists_1_end>0.01,dists_2_end>0.01)
            dist_end_center_mask = torch.logical_and(dist_end_mask, center_dist>0.005)
            no_collision_mask_o3d = torch.logical_and(no_collision_mask_o3d,dist_end_center_mask)
        geometry_mask = torch.logical_and(geometry_mask, no_collision_mask_o3d)
    else:
        # use cdist
        dists_1 = torch.min(torch.cdist(side_points_1, pos, p=2), dim=-1, keepdim=False)[0]
        dists_2 = torch.min(torch.cdist(side_points_2, pos, p=2), dim=-1, keepdim=False)[0]
        no_collision_mask_cdist = torch.logical_and(dists_1 > 1e-2, dists_2 > 1e-2)
        if strict:
            dists_1_end = torch.min(torch.cdist(side_points_1_end, pos, p=2), dim=-1, keepdim=False)[0]
            dists_2_end = torch.min(torch.cdist(side_points_2_end, pos, p=2), dim=-1, keepdim=False)[0]
            center_dists = torch.min(torch.cdist(gripper_center, pos, p=2), dim=-1, keepdim=False)[0]
            dist_end_mask = torch.logical_and(dists_1_end > 0.01, dists_2_end > 0.01)
            dist_end_center_mask = torch.logical_and(dist_end_mask, center_dists > 0.005)
            no_collision_mask_cdist = torch.logical_and(no_collision_mask_cdist, dist_end_center_mask)
        geometry_mask = torch.logical_and(geometry_mask, no_collision_mask_cdist)
    return geometry_mask,depth_projection,orth_mask,angle_mask,half_baseline_projection


def points2pcd(points,normals=None,vis=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    if vis:
        o3d.visualization.draw_geometries([pcd])
    return pcd

def vis_samples_2(cloud, pos, neg):
    '''
    function to visualize the pcd
    :param cloud: open3d instance
    :param pos: the point indices that will be colored in red
    :param neg: the point indices that will be colored in green
    :return:
    '''

    inlier1_cloud = cloud.select_by_index(pos)

    if len(neg)>0:
        inlier2_cloud = cloud.select_by_index(neg)
        inlier2_cloud.paint_uniform_color([1, 0, 0])
        ind = np.concatenate((pos, neg))
    else:
        ind = pos
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    inlier1_cloud.paint_uniform_color([0, 1, 0])
    if len(neg)>0:
        o3d.visualization.draw_geometries([inlier1_cloud, inlier2_cloud, outlier_cloud],)
    else:
        o3d.visualization.draw_geometries([inlier1_cloud, outlier_cloud],)

def orthognal_grasps(geometry_mask,depth_projection,sample_normal,des_normals,sample_pos):

    '''
    :param geometry_mask: [bool,bool,,]
    :param depth_projection:
    :param sample_normal:
    :param des_normals:
    :param sample_pos:
    :return: mX4X4 matrices that used to execute grasp in simulation
    '''
    # if these is no reasonable points do nothing
    assert sum(geometry_mask)>0
    depth = depth_projection[geometry_mask]
    # finger depth
    gripper_dis_from_source = (0.072-0.007 - depth).unsqueeze(dim=-1)
    z_axis = -sample_normal[geometry_mask]  # todo careful
    y_axis = des_normals[geometry_mask]
    x_axis = torch.cross(y_axis, z_axis,dim=1)
    x_axis = F.normalize(x_axis, p=2,dim=1)
    y_axis = torch.cross(z_axis, x_axis,dim=1)
    y_axis = F.normalize(y_axis, p=2,dim=1)
    gripper_position = gripper_dis_from_source.repeat(1, 3) * (-z_axis) + sample_pos[geometry_mask]
    transform_matrix = torch.cat((x_axis.unsqueeze(dim=-1), y_axis.unsqueeze(dim=-1),
                                  z_axis.unsqueeze(dim=-1), gripper_position.unsqueeze(dim=-1)), dim=-1)
    homo_agument = torch.as_tensor([0., 0., 0., 1.]).reshape(1, 1, 4).repeat(len(z_axis), 1, 1)
    transform_matrix = torch.cat((transform_matrix, homo_agument), dim=1)
    #transform_matrix = transform_matrix.numpy()
    #print(transform_matrix.shape)

    # flip_trans = torch.as_tensor([[1.0, 0., 0., 0.],
    #                               [0.0, -1., 0., 0.],
    #                               [0.0, 0., -1., 0.],
    #                               [0.0, 0., 0., 1.]])
    # transform_matrix = torch.einsum('nij,jk->nik', transform_matrix, flip_trans)
    return transform_matrix

def draw_grasps(geometry_mask,depth_projection,sample_normal,des_normals,sample_pos,radius_p_index,sample,pos,diverse=False,scores=None):
    '''
    functions to visualize the grasp poses and the point cloud as well the sample points and the contact points
    :param geometry_mask:
    :param depth_projection:
    :param sample_normal:
    :param des_normals:
    :param sample_pos:
    :param radius_p_index:
    :param sample:
    :param pos:
    :param diverse:
    :param scores:
    :return:
    '''
    # if these is no reasonable points do nothing
    assert sum(geometry_mask)>0
    depth = depth_projection[geometry_mask]
    gripper_dis_from_source = (0.105 - depth).unsqueeze(dim=-1)
    z_axis = sample_normal[geometry_mask]  # todo careful
    x_axis = des_normals[geometry_mask]
    y_axis = torch.cross(z_axis, x_axis,dim=1)
    y_axis = F.normalize(y_axis,p=2,dim=1)
    x_axis = torch.cross(y_axis, z_axis,dim=1)
    x_axis = F.normalize(x_axis, p=2, dim=1)
    gripper_position = gripper_dis_from_source.repeat(1, 3) * z_axis + sample_pos[geometry_mask]
    transform_matrix = torch.cat((x_axis.unsqueeze(dim=-1), y_axis.unsqueeze(dim=-1),
                                  z_axis.unsqueeze(dim=-1), gripper_position.unsqueeze(dim=-1)), dim=-1)
    homo_agument = torch.as_tensor([0., 0., 0., 1.]).reshape(1, 1, 4).repeat(len(z_axis), 1, 1)
    transform_matrix = torch.cat((transform_matrix, homo_agument), dim=1)
    flip_trans = torch.as_tensor([[1.0, 0., 0., 0.],
                                  [0.0, -1., 0., 0.],
                                  [0.0, 0., -1., 0.],
                                  [0.0, 0., 0., 1.]])
    transform_matrix = torch.einsum('nij,jk->nik', transform_matrix.double(), flip_trans.double())
    pc_colors = np.tile(np.asarray([[0.1, 0.1, 1.]]), (len(pos), 1))
    pc_colors[radius_p_index[geometry_mask].numpy(), :] = np.asarray([0.1, 1, 0.1])
    pc_colors[sample, :] = np.asarray([1.0, 0.1, 0.1])
    draw_scene(pc = pos.numpy(), grasps=transform_matrix.numpy(), show_gripper_mesh=False, visualize_diverse_grasps=diverse,pc_color=pc_colors,grasp_scores=scores)
    mlab.show()

def draw_single_grasp(geometry_mask,depth_projection,sample_normal,des_normals,sample_pos,sample,pos,idx,diverse=False,scores=None):
    '''
    similar function to draw grasp,  draw a single grasp and the point cloud.
    This function is used to have an understanding when grasp fails.
    '''
    # if these is no reasonable points do nothing
    assert sum(geometry_mask)>0
    depth = depth_projection[geometry_mask]
    gripper_dis_from_source = (0.105 - depth).unsqueeze(dim=-1)
    z_axis = sample_normal[geometry_mask]  # todo careful
    x_axis = des_normals[geometry_mask]
    y_axis = torch.cross(z_axis, x_axis,dim=1)
    y_axis = F.normalize(y_axis, p=2, dim=1)
    x_axis = torch.cross(y_axis, z_axis,dim=1)
    x_axis = F.normalize(x_axis, p=2, dim=1)
    gripper_position = gripper_dis_from_source.repeat(1, 3) * z_axis + sample_pos[geometry_mask]
    transform_matrix = torch.cat((x_axis.unsqueeze(dim=-1), y_axis.unsqueeze(dim=-1),
                                  z_axis.unsqueeze(dim=-1), gripper_position.unsqueeze(dim=-1)), dim=-1)
    homo_agument = torch.as_tensor([0., 0., 0., 1.]).reshape(1, 1, 4).repeat(len(z_axis), 1, 1)
    transform_matrix = torch.cat((transform_matrix, homo_agument), dim=1)
    flip_trans = torch.as_tensor([[1.0, 0., 0., 0.],
                                  [0.0, -1., 0., 0.],
                                  [0.0, 0., -1., 0.],
                                  [0.0, 0., 0., 1.]])
    transform_matrix = torch.einsum('nij,jk->nik', transform_matrix.double(), flip_trans.double())
    pc_colors = np.tile(np.asarray([[0.1, 0.1, 1.]]), (len(pos), 1))
    #pc_colors[radius_p_index[geometry_mask].numpy(), :] = np.asarray([0.1, 1, 0.1])
    pc_colors[sample, :] = np.asarray([1.0, 0.1, 0.1])
    transform_matrix=transform_matrix.numpy()[idx,:,:]
    print(transform_matrix.shape)
    print(transform_matrix)
    print(np.sum(np.power(transform_matrix,2),axis=0))
    x = transform_matrix[:, 0]
    y = transform_matrix[:, 1]
    z = transform_matrix[:, 2]
    print(np.dot(x, y), np.dot(y, z), np.dot(x, z))
    draw_scene(pc = pos.numpy(), grasps=[transform_matrix], show_gripper_mesh=False, visualize_diverse_grasps=diverse,pc_color=pc_colors,grasp_scores=scores)
    mlab.show()