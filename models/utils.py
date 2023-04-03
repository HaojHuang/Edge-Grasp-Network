import os.path
import torch
import numpy as np

def get_geometry_mask(normals_dot,dot_product_2,relative_pos,des_normals,sample_normal,sample_pos,pos,strict=False):
    orth_mask = abs(normals_dot) < 0.1
    pitch_constrain_mask = abs(dot_product_2) < 0.1
    orth_mask = orth_mask.squeeze(dim=-1)
    pitch_constrain_mask = pitch_constrain_mask.squeeze(dim=-1)
    angle_mask = torch.logical_and(orth_mask,pitch_constrain_mask)
    half_baseline_projection = torch.sum(relative_pos * des_normals, dim=-1)
    depth_projection = -torch.sum(relative_pos * sample_normal, dim=-1)
    geometry_mask_1 = torch.logical_and(0.003 < half_baseline_projection, half_baseline_projection < 0.038)
    geometry_mask_2 = torch.logical_and(0 < depth_projection, depth_projection < 0.043)
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
    return geometry_mask,depth_projection,orth_mask,angle_mask

def get_geometry_mask2(normals_dot,dot_product_2,relative_pos,des_normals,sample_normal,sample_pos,pos,strict=False):
    orth_mask = abs(normals_dot) < 0.1
    pitch_constrain_mask = abs(dot_product_2) < 0.1
    orth_mask = orth_mask.squeeze(dim=-1)
    pitch_constrain_mask = pitch_constrain_mask.squeeze(dim=-1)
    angle_mask = torch.logical_and(orth_mask,pitch_constrain_mask)
    half_baseline_projection = torch.sum(relative_pos * des_normals, dim=-1)
    depth_projection = -torch.sum(relative_pos * sample_normal, dim=-1)
    geometry_mask_1 = torch.logical_and(0.003 < half_baseline_projection, half_baseline_projection < 0.038)
    geometry_mask_2 = torch.logical_and(0 < depth_projection, depth_projection < 0.043)
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
    return geometry_mask,depth_projection,orth_mask,angle_mask,no_collision_mask_cdist,pitch_constrain_mask

def create_csv(path, columns):
    with open(path,"w") as f:
        f.write(",".join(columns))
        f.write("\n")

def append_csv(path, *args):
    row = ",".join([str(arg) for arg in args])
    with open(path,"a") as f:
        f.write(row)
        f.write("\n")

def write_training(root, epoch,step,loss,train_accu,train_balanced_accu,):
    # TODO concurrent writes could be an issue
    csv_path = os.path.join(root,"training.csv")
    if not os.path.exists(csv_path):
        create_csv(
            csv_path,
            ["epoch", "step", "loss", "train_accu", "train_balanced_accu"],
        )
    append_csv(csv_path,epoch,step,loss,train_accu,train_balanced_accu,)

def write_test(root, epoch,step,loss,test_accu,test_balanced_accu,):
    # TODO concurrent writes could be an issue
    csv_path = os.path.join(root,"test.csv")
    if not os.path.exists(csv_path):
        create_csv(
            csv_path,
            ["epoch", "step", "loss", "test_accu","test_balanced_accu"],
        )
    append_csv(csv_path,epoch,step,loss,test_accu,test_balanced_accu,)


def write_test2(root, epoch,step,loss_total,loss,test_accu,test_balanced_accu,loss_dot2,test_accu_dot2,test_balanced_accu_dot2,):
    # TODO concurrent writes could be an issue
    csv_path = os.path.join(root,"test.csv")
    if not os.path.exists(csv_path):
        create_csv(
            csv_path,
            ["epoch", "step", "loss_total", "loss","test_accu","test_balanced_accu", "loss_dot2","test_accu_dot2","test_balanced_accu_dot2"],
        )
    append_csv(csv_path,epoch,step,loss_total,loss,test_accu,test_balanced_accu,loss_dot2,test_accu_dot2,test_balanced_accu_dot2)


def write_test3(root, epoch,step,loss_total,loss,test_accu,test_balanced_accu,
                loss2,test_accu2,test_balanced_accu2,
                loss3,test_accu3,test_balanced_accu3,):
    # TODO concurrent writes could be an issue
    csv_path = os.path.join(root,"test.csv")
    if not os.path.exists(csv_path):
        create_csv(
            csv_path,
            ["epoch", "step", "loss_total", "loss","test_accu","test_balanced_accu",
             "loss2","test_accu2","test_balanced_accu2",
             "loss3","test_accu3","test_balanced_accu3"],
        )
    append_csv(csv_path,epoch,step,loss_total,loss,test_accu,test_balanced_accu,
               loss2,test_accu2,test_balanced_accu2,
               loss3,test_accu3,test_balanced_accu3,)