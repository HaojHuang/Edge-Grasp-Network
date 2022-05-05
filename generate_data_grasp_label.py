import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import scipy.signal as signal
from grasp import Grasp, Label
from perception import *
from simulation_float_generate_data import ClutterRemovalSim
from transform import Rotation, Transform
from io_smi import *
from utility import FarthestSampler, get_geometry_mask, vis_samples_2, orthognal_grasps, draw_grasps, draw_single_grasp
import warnings
import torch
from torch_geometric.nn import radius
import torch.nn.functional as F

warnings.filterwarnings("ignore")
# OBJECT_COUNT_LAMBDA = 4
# MAX_VIEWPOINT_COUNT = 4

GRASPS_POINT_SCENE = 32
NUMBER_SCENE = 300
NUMBER_VIEWS = 2


def main(args):
    sim = ClutterRemovalSim(args.scene, args.object_set, gui=args.sim_gui)
    finger_depth = sim.gripper.finger_depth
    root = Path('data_robot/raw/foo6')
    (root / "pcd").mkdir(parents=True, exist_ok=True)
    write_setup(
        root,
        sim.size,
        sim.camera.intrinsic,
        sim.gripper.max_opening_width,
        sim.gripper.finger_depth, )

    success_grasp_totoal = 0 # of successful grasp
    orthognal_grasp = 0 # of orthogonal grasp
    orthognal_point_num = 0 # of points which has orthogonal neighbors
    point_num = 0 #of points sampled
    success_ortho_point_num = 0
    pre = 0 #grasp collision when approach the target pose
    after = 0 #object slide
    grasp_hit = 0 # collision during grasp?  a little weired
    des_success_total = 0
    success_object = 0
    total_objects = 0

    for scene_num in range(NUMBER_SCENE):
        object_count = 1
        sim.reset(object_count)
        sim.save_state()
        # render synthetic depth images
        object_orthogonal_flag = False
        success_object_flag = False
        for partial_view in range(NUMBER_VIEWS):
            n = 1
            depth_imgs, extrinsics, eye = render_images(sim, n)
            # reconstrct point cloud using a subset of the images
            tsdf = create_tsdf(sim.size, 180, depth_imgs, sim.camera.intrinsic, extrinsics)
            pc = tsdf.get_cloud()
            camera_location = eye
            # crop surface and borders from point cloud
            bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
            # o3d.visualization.draw_geometries([pc])
            pc = pc.crop(bounding_box)
            if pc.is_empty():
                print("Empty point cloud, skipping scene")
                continue

            if args.add_noise:
                vertices = np.asarray(pc.points)
                # add gaussian noise 95% confident interval (-1.96,1.96)
                vertices = vertices + np.random.normal(loc=0.0, scale=0.001, size=(len(vertices), 3))
                pc.points = o3d.utility.Vector3dVector(vertices)

            pc, ind = pc.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
            pc, ind = pc.remove_radius_outlier(nb_points=30, radius=0.03)
            pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pc.orient_normals_consistent_tangent_plane(30)
            pc.orient_normals_towards_camera_location(camera_location=camera_location)
            normals = np.asarray(pc.normals)
            # direction = -np.asarray(pc.points) + camera_location
            # dot = np.sum(normals * direction, axis=-1)
            # mask = dot < -0.0
            # normals[mask] *= -1
            vertices = np.asarray(pc.points)
            if len(vertices) < 200:
                continue
            if len(vertices) > 1024:
                sampler = FarthestSampler()
                vertices, idx = sampler(vertices, 1024)
                normals = normals[idx]
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(vertices)
                pc.normals = o3d.utility.Vector3dVector(normals)

            #o3d.visualization.draw_geometries([pc], point_show_normal=True)
            #break
            pos = np.asarray(pc.points)
            normals = np.asarray(pc.normals)
            pos = torch.from_numpy(pos)
            normals = torch.from_numpy(normals)
            # print(pos.shape,normals.shape)
            for _ in range(1):
                sample_number = 32
                sample = np.random.choice(len(pos), sample_number,replace=False)
                sample = np.unique(sample)
                sample_pos = pos[sample, :]
                sample_normal = normals[sample, :]
                radius_p_batch_index = radius(pos, sample_pos, r=0.038, max_num_neighbors=1024)
                radius_p_index = radius_p_batch_index[1, :]
                radius_p_batch = radius_p_batch_index[0, :]
                sample_pos = torch.cat(
                    [sample_pos[i, :].repeat((radius_p_batch == i).sum(), 1) for i in range(len(sample))],
                    dim=0)
                sample_normal = torch.cat(
                    [sample_normal[i, :].repeat((radius_p_batch == i).sum(), 1) for i in range(len(sample))], dim=0)
                des_pos = pos[radius_p_index, :]
                des_normals = normals[radius_p_index, :]
                normals_dot = torch.einsum('ik,ik->i', des_normals, sample_normal).unsqueeze(dim=-1)
                relative_pos = des_pos - sample_pos
                relative_pos_normalized = F.normalize(relative_pos, p=2, dim=1)
                third_axis = torch.cross(relative_pos_normalized, sample_normal, dim=1)
                # todo: after cross, the norm is changed
                #print('third axis',third_axis.size(),third_axis)
                third_axis = F.normalize(third_axis,p=2,dim=1)
                #print(third_axis.size(),third_axis)
                dot_product_2 = torch.einsum('ik,ik->i', des_normals, third_axis).unsqueeze(dim=-1)
                geometry_mask, depth_projection, orth_mask, angle_mask, width = get_geometry_mask(normals_dot, dot_product_2,
                                                                                           relative_pos, des_normals,
                                                                                           sample_normal, sample_pos, pos,
                                                                                           use_o3d=True, strict=True)

                # print(geometry_mask.size())
                # print(geometry_mask)
                point_num += 1
                if sum(orth_mask)>0:
                    #vis_samples_2(pc,sample, radius_p_index[orth_mask].numpy())
                    #vis_samples_2(pc, sample, radius_p_index[angle_mask].numpy())
                    #vis_samples_2(pc, sample, radius_p_index[geometry_mask].numpy())
                    pass
                if sum(geometry_mask) > 0:
                    object_orthogonal_flag = True
                    trans_matrix = orthognal_grasps(geometry_mask, depth_projection, sample_normal, des_normals, sample_pos)
                    width = width[geometry_mask].numpy() * 2 + 0.06 #just use the max_gripper_width
                    res = evaluate_grasps(sim,poses=trans_matrix.numpy(),gripper_widths=width)
                    success_num, des_list = res
                    grasp_collison_label = np.zeros(len(des_list))
                    grasp_drop_label = np.zeros(len(des_list))

                    for idx,des in enumerate(des_list):
                        if des == 'pregrasp':
                            pre +=1
                        if des == 'grasp':
                            grasp_hit +=1
                            grasp_collison_label[idx] = 1
                        if des == 'after':
                            after +=1
                            grasp_drop_label[idx] =1
                        if des == 'success':
                            des_success_total +=1

                    write_edge_data(root, pos=pos.numpy(), normals=normals.numpy(), sample=sample, radius_p_index=radius_p_index.numpy(),
                                    radius_p_batch=radius_p_batch.numpy(), dot1=normals_dot.numpy(), dot2=dot_product_2.numpy(),
                                    geometry_mask=geometry_mask.numpy(), grasp_label=np.asarray(success_num),
                                    grasp_collison_label = grasp_collison_label, grasp_drop_label = grasp_drop_label,
                                    depth_projection=depth_projection.numpy(), angle_mask = angle_mask.numpy(),)
                    # print(sum(grasp_collison_label)+sum(grasp_drop_label) + sum(success_num))
                    # print(sum(geometry_mask))
                    #print(des_list)
                    success_grasp_totoal += sum(success_num)
                    orthognal_grasp += len(trans_matrix)
                    orthognal_point_num += 1
                    if sum(success_num)>0:
                        success_ortho_point_num += 1
                        success_object_flag = True
                else:
                    continue

        if success_object_flag:
            success_object += 1
        if object_orthogonal_flag:
            total_objects += 1

        print('Success Object {}, Objects {}, '
              'Success Orth Views {}, Orth Views {}, Total views {}, '
              'Success Grasp {}, Total Grasp {}, '
              'Pre Collision {}, Failed during Grasping {}, Drop {}, Success_Des {}'
              .format(success_object,total_objects,
                      success_ortho_point_num,orthognal_point_num,point_num,
                      success_grasp_totoal,orthognal_grasp,
                      pre,grasp_hit,after,des_success_total))
            # store the raw data

def render_images(sim, n):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0+0.25]) #0.25 is the floating z value
    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)
    eye =None
    for i in range(n):
        r = np.random.uniform(1.0, 1.5) * sim.size
        theta = np.random.uniform(0.0, np.pi/3)
        phi = np.random.uniform(0.0, 2.0 * np.pi)
        extrinsic = camera_on_sphere(origin, r, theta, phi)
        depth_img = sim.camera.render(extrinsic)[1]
        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img
        eye = np.r_[
            r * sin(theta) * cos(phi),
            r * sin(theta) * sin(phi),
            r * cos(theta),
        ]
        eye = eye + origin.translation
    return depth_imgs, extrinsics, eye


def normalization(x):
    x_norm = max(np.linalg.norm(x), 1e-12)
    x = x / x_norm
    return x


def evaluate_grasps(sim,poses, gripper_widths):
    outcomes, widths, describtions = [], [], []
    for i in range(len(poses)):
        pose = poses[i,:,:]
        sim.restore_state()
        gripper_width = min(sim.gripper.max_opening_width,gripper_widths[i])
        candidate = Grasp(Transform.from_matrix(pose), width=gripper_width)
        outcome, width, describtion = sim.execute_grasp(candidate, remove=False)
        sim.restore_state()
        outcomes.append(outcome)
        widths.append(width)
        describtions.append(describtion)
        #break
    successes = (np.asarray(outcomes) == Label.SUCCESS).astype(int)
    return successes,describtions


def write_edge_data(root, pos, normals, sample, radius_p_index, radius_p_batch, dot1, dot2, geometry_mask, grasp_label,
                    grasp_collison_label, grasp_drop_label, depth_projection, angle_mask):
    scene_id = uuid.uuid4().hex
    path = root / "pcd" / (scene_id + ".npz")
    np.savez_compressed(path,pos=pos, normals=normals, sample=sample, radius_p_index=radius_p_index,
                        radius_p_batch=radius_p_batch, dot1=dot1, dot2=dot2, geometry_mask=geometry_mask, grasp_label=grasp_label,
                        grasp_collison_label=grasp_collison_label, grasp_drop_label=grasp_drop_label,
                        depth_projection=depth_projection, angle_mask=angle_mask,
                        )
    return scene_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="pile")
    parser.add_argument("--object-set", type=str, default="pile/train")
    parser.add_argument("--num-grasps", type=int, default=10000)
    parser.add_argument("--sim-gui", action="store_true", default=False)
    parser.add_argument("--add_noise", action="store_true", default=False)
    args = parser.parse_args()
    main(args)