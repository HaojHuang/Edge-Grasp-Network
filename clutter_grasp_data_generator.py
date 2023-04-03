import argparse
from pathlib import Path
import numpy
import numpy as np
import open3d as o3d
import scipy.signal as signal
from simulator.grasp import Grasp, Label
from simulator.simulation_clutter_bandit import ClutterRemovalSim
# from simulator.transform import Rotation, Transform
from simulator.io_smi import *
from simulator.utility import FarthestSampler, vis_samples_2, orthognal_grasps
import warnings
import torch
from torch_geometric.nn import radius
import torch.nn.functional as F
from simulator.utility import FarthestSamplerTorch
# from torch_geometric.data import Data, Batch
# from torch_scatter import scatter
# from torch.backends import cudnn
warnings.filterwarnings("ignore")


RUN_TIMES = 1
#GRASPS_TRIAL_SCENE = 1000
NUMBER_SCENE = 100
NUMBER_VIEWS = 5
SAMPLE_PER_ANCHOR = 150
OBJECT_COUNT_LAMBDA=4

def main(args):
    device = args.device
    if device == 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')
    sim = ClutterRemovalSim(args.scene, args.object_set, gui=args.sim_gui)
    finger_depth = sim.gripper.finger_depth
    root = Path('./raw_data')
    (root / "pcd").mkdir(parents=True, exist_ok=True)
    write_setup(
        root,
        sim.size,
        sim.camera.intrinsic,
        sim.gripper.max_opening_width,
        sim.gripper.finger_depth, )
    #sim.show_object()
    recoder_level0 = []
    for RUN in range(RUN_TIMES):
        # np.random.seed(RUN+1)
        # torch.set_num_threads(RUN+1)
        # torch.manual_seed(RUN+1)
        # cudnn.benchmark = True
        # cudnn.deterministic = True
        recoder_level1 = []
        success_grasp_totoal = 0 # of successful grasp
        orthognal_grasp = 0 # of total grasp
        pre = 0 #grasp collision when approach the target pose
        after = 0 #object slide
        grasp_hit = 0 # collision during grasp?  a little weired
        des_success_total = 0
        success_object = 0
        total_objects = 0
        no_candidate_object = 0
        for scene_num in range(NUMBER_SCENE):
            object_count = np.random.poisson(OBJECT_COUNT_LAMBDA) + 1
            sim.reset(object_count)
            sim.save_state()
            # render synthetic depth images
            success_object_flag = False
            total_objects += 1
            no_candidate_time = 0
            for partial_view in range(NUMBER_VIEWS):
                n = np.random.choice(a=[1, 2, 3], p=[0.6, 0.2, 0.2])
                depth_imgs, extrinsics, eye = render_images(sim, n)
                # reconstrct point cloud using a subset of the images
                tsdf = create_tsdf(sim.size, 180, depth_imgs, sim.camera.intrinsic, extrinsics)
                pc = tsdf.get_cloud()
                camera_location = eye
                # crop surface and borders from point cloud
                bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
                pc = pc.crop(bounding_box)
                if pc.is_empty():
                    print("Empty point cloud, skipping scene")
                    continue
                if args.add_noise:
                    vertices = np.asarray(pc.points)
                    # add gaussian noise 95% confident interval (-1.96,1.96)
                    vertices = vertices + np.random.normal(loc=0.0,scale=0.0005, size=(len(vertices),3))
                    pc.points = o3d.utility.Vector3dVector(vertices)

                pc, ind = pc.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
                pc, ind = pc.remove_radius_outlier(nb_points=30, radius=0.03)
                pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                pc.orient_normals_consistent_tangent_plane(20)
                if n ==1:
                    pc.orient_normals_towards_camera_location(camera_location=camera_location)
                # normals = np.asarray(pc.normals)
                # direction = -np.asarray(pc.points) + camera_location
                # dot = np.sum(normals * direction, axis=-1)
                # mask = dot < -0.0
                # normals[mask] *= -1
                vertices = np.asarray(pc.points)
                if len(vertices) < 300:
                    print("point cloud<300, skipping scene")
                    continue
                pc = pc.voxel_down_sample(voxel_size=0.0045)
                # visualize the point cloud
                #o3d.visualization.draw_geometries([pc])
                pos = np.asarray(pc.points)
                normals = np.asarray(pc.normals)
                pos = torch.from_numpy(pos).to(torch.float32)
                normals = torch.from_numpy(normals).to(torch.float32)
                sample_number = args.sample_number
                sample = np.random.choice(len(pos), sample_number, replace=False)
                sample = torch.from_numpy(sample).to(torch.long)
                sample_pos = pos[sample, :]

                #####
                # virtul_normal = torch.rand(sample_number, 3)
                # virtul_normal = F.normalize(virtul_normal, p=2, dim=1)
                # sample_normal = virtul_normal
                # print(approach_vector)
                #print('approach direction',_order,)

                radius_p_batch_index = radius(pos, sample_pos, r=0.038, max_num_neighbors=1024)
                radius_p_index = radius_p_batch_index[1, :]
                radius_p_batch = radius_p_batch_index[0, :]
                sample_pos = torch.cat(
                    [sample_pos[i, :].repeat((radius_p_batch == i).sum(), 1) for i in range(len(sample))],
                    dim=0)
                sample_copy = sample.clone().unsqueeze(dim=-1)
                sample_index = torch.cat(
                    [sample_copy[i, :].repeat((radius_p_batch == i).sum(), 1) for i in range(len(sample))], dim=0)
                edges = torch.cat((sample_index,radius_p_index.unsqueeze(dim=-1)),dim=1)
                all_edge_index = numpy.arange(0,len(edges))
                all_edge_index = torch.from_numpy(all_edge_index).to(torch.long)
                des_pos = pos[radius_p_index, :]
                des_normals = normals[radius_p_index, :]
                relative_pos = des_pos - sample_pos
                relative_pos_normalized = F.normalize(relative_pos, p=2, dim=1)
                # sample approach vector
                label_record = []
                edge_sample_index = []
                quat_record = []
                translation_record = []
                # only record approach vectors with a angle mask
                x_axis = torch.cross(des_normals, relative_pos_normalized)
                x_axis = F.normalize(x_axis, p=2, dim=1)
                valid_edge_approach = torch.cross(x_axis, des_normals)
                valid_edge_approach = F.normalize(valid_edge_approach, p=2, dim=1)
                valid_edge_approach = -valid_edge_approach
                # print('new approachs',valid_edge_approach.shape)
                up_dot_mask = torch.einsum('ik,k->i', valid_edge_approach, torch.tensor([0., 0., 1.]))
                # print(up_dot_mask.size())
                relative_norm = torch.linalg.norm(relative_pos, dim=-1)
                depth_projection = -torch.sum(relative_pos * valid_edge_approach, dim=-1)
                # print(relative_norm.size())
                # geometry_mask, depth_projection, half_baseline_projection = get_edge_nocolli_mask(relative_pos,
                #                                                                                   des_normals,
                #                                                                                   valid_edge_approach,
                #                                                                                   sample_pos,pos,use_o3d=True,strict=True)
                fps_torch = FarthestSamplerTorch()
                # fps_numpy = FarthestSampler()
                edge_global_idx = []
                for i in range(args.sample_number):
                    batch_mask = radius_p_batch == i
                    num = torch.sum(batch_mask).item()
                    if num < SAMPLE_PER_ANCHOR:
                        edge_global_idx.append(all_edge_index[batch_mask])
                    else:
                        des_batch_pos = des_pos[batch_mask, :]
                        des_batch_index = all_edge_index[batch_mask]
                        _, idx = fps_torch(des_batch_pos, SAMPLE_PER_ANCHOR)
                        edge_global_idx.append(des_batch_index[idx])
                        #print(idx,des_batch_index[idx])
                #print(edge_global_idx)
                edge_global_idx = torch.cat(edge_global_idx, dim=0).reshape(-1)
                #print(edge_global_idx.size())
                geometry_mask = torch.ones(len(edges))>2
                geometry_mask[edge_global_idx] = True
                geometry_mask = torch.logical_and(geometry_mask, relative_norm>0.003)
                geometry_mask = torch.logical_and(up_dot_mask > -0.1, geometry_mask)

                if torch.sum(geometry_mask).item()>0:
                    trans = get_grasp_poses(geometry_mask,depth_projection,approaches=valid_edge_approach,
                                            des_normals=des_normals,sample_pos=sample_pos,x_axis=x_axis)
                    no_collision_table_mask = get_gripper_points_mask(trans,threshold=0.06)
                    if torch.sum(no_collision_table_mask).item()>0:
                        geometry_mask[geometry_mask==True]=no_collision_table_mask
                        assert torch.sum(geometry_mask).item()==torch.sum(no_collision_table_mask).item()
                    else:
                        continue
                else:
                    continue

                edge_sample_index = all_edge_index[geometry_mask]
                #depth_projection = -torch.sum(relative_pos * valid_edge_approach, dim=-1)
                #print(depth_projection)
                #depth_projection_mask = torch.logical_and(depth_projection>0,depth_projection<0.035)
                #geometry_mask = torch.logical_and(geometry_mask,depth_projection_mask)
                print('{} grasps without colliding table of {} total samples'.format(torch.sum(geometry_mask).item(),len(geometry_mask)))
                if torch.sum(geometry_mask).item() > 0:
                    # visualize sampled points: green->approach point, red->contact
                    # vis_samples_2(pc, sample.numpy(), radius_p_index[geometry_mask].numpy())
                    pass
                if len(edge_sample_index)>0:
                    if len(edge_sample_index) > 2000:
                        edge_sample_index = edge_sample_index[torch.randperm(len(edge_sample_index))[:2000]]
                        edge_sample_index,_ = torch.sort(edge_sample_index)
                        geometry_mask = torch.ones(len(depth_projection)) > 2.
                        geometry_mask[edge_sample_index] = True
                    #print('grasps per scene', sum(geometry_mask))
                    trans_matrix = orthognal_grasps(geometry_mask, depth_projection, valid_edge_approach, des_normals,sample_pos)
                    # half_widthes = torch.abs(torch.sum(relative_pos[geometry_mask,:]*des_normals[geometry_mask,:],dim=1))+0.013
                    # widthes = (half_widthes*2.).clip(max=0.08)
                    # print(widthes)
                else:
                    print('No candidate')
                    no_candidate_time += 1
                    continue
                    # evaluation
                res = evaluate_grasps(sim,poses=trans_matrix.numpy(),widthes=None)
                success_num, des_list,quats, translations = res
                quat_record.extend(quats)
                translation_record.extend(translations)
                label_record.extend(success_num)
                success_grasp_totoal += sum(success_num)
                orthognal_grasp += len(trans_matrix)
                for idx,des in enumerate(des_list):
                    if des == 'pregrasp':
                        pre +=1
                    if des == 'grasp':
                        grasp_hit +=1
                    if des == 'after':
                        after +=1
                    if des == 'success':
                        des_success_total +=1
                        success_object_flag = True

                # print('label_record', label_record)
                # print('approach vector',len(approach_record))
                # print('sample edges',len(edge_sample_index))
                # print('quats',len(quat_record))
                # print('translation',len(translation_record))
                # print('sample per app', len(samples_per_approach))
                write_implict_data(root, pos=pos.numpy(), normals=normals.numpy(), sample=sample, radius_p_index=radius_p_index.numpy(),
                                   radius_p_batch=radius_p_batch.numpy(),grasp_label=np.asarray(label_record),
                                   edges=edges.numpy(), approachs=valid_edge_approach[geometry_mask,:],depth_projection=depth_projection[geometry_mask],
                                   edge_sample_index=numpy.asarray(edge_sample_index), quat_record=numpy.asarray(quat_record),
                                   translation_record=np.asarray(translation_record),)

            if success_object_flag:
                success_object += 1
            if no_candidate_time==NUMBER_VIEWS:
                no_candidate_object +=1
            print('Success Object {}, No candidate Object {}, Objects {}, '
                  'Success Grasp {}, Total Grasp {}, '
                  'Pre Collision {}, Failed during Grasping {}, Drop {}, Success_Des {}, '
                  .format(success_object,no_candidate_object, total_objects,
                          success_grasp_totoal,orthognal_grasp,
                          pre,grasp_hit,after,des_success_total,))
            # recoder_level1.append([success_object,no_candidate_object, total_objects,
            #                        success_grasp_totoal,orthognal_grasp,pre,grasp_hit,after,des_success_total])
            # np.save('recoder_level1'+str(args.scene),np.asarray(recoder_level1))
        # recoder_level0.append(recoder_level1)
        # np.save('recoder_level0'+str(args.scene), np.asarray(recoder_level0))


def render_images(sim, n):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0 + 0.25])
    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)
    for i in range(n):
        r = np.random.uniform(1.5, 2.5) * sim.size
        theta = np.random.uniform(np.pi/4, np.pi/3)
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

def sample_trig(a_c):
    theta = np.random.rand(a_c) * np.pi * 2
    v = np.random.uniform(0.5, 1, size=a_c)
    phi = np.arccos(2 * v - 1)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    x = cos_theta * sin_phi
    y = sin_theta * sin_phi
    z = cos_phi
    vectors = np.stack((x, y, z), axis=1)
    return vectors

def normalization(x):
    x_norm = max(np.linalg.norm(x), 1e-12)
    x = x / x_norm
    return x

def evaluate_grasps(sim,poses,widthes=None):
    outcomes, widths, describtions = [], [], []
    quats, translations = [],[]
    for i in range(len(poses)):
        pose = poses[i,:,:]
        sim.restore_state()
        dof_6 = Transform.from_matrix(pose)
        # decompose the quat
        quat = dof_6.rotation.as_quat()
        translation = dof_6.translation
        if widthes is not None:
            width = widthes[i]
        else:
            width = sim.gripper.max_opening_width
        candidate = Grasp(dof_6, width=width)
        outcome, width, describtion = sim.execute_grasp(candidate, remove=False)
        sim.restore_state()
        outcomes.append(outcome)
        widths.append(width)
        describtions.append(describtion)
        quats.append(quat)
        translations.append(translation)
        #break
    successes = (np.asarray(outcomes) == Label.SUCCESS).astype(int)
    return successes,describtions,quats,translations


def write_implict_data(root, pos, normals, sample, radius_p_index, radius_p_batch, edges,depth_projection, approachs,
                       edge_sample_index, grasp_label, quat_record, translation_record,):
    scene_id = uuid.uuid4().hex
    path = root / "pcd" / (scene_id + ".npz")
    np.savez_compressed(path, pos=pos, normals=normals, sample=sample, radius_p_index=radius_p_index,
                        radius_p_batch=radius_p_batch, edges=edges,depth_projection=depth_projection,
                        approachs=approachs, edge_sample_index=edge_sample_index, grasp_label=grasp_label,
                        quat=quat_record, translation=translation_record,)
    return scene_id

def get_grasp_poses(geometry_mask, depth_projection, approaches, des_normals, sample_pos, x_axis):

    '''
    :param geometry_mask: [bool,bool,,]
    :param depth_projection:
    :param sample_normal:
    :param des_normals:
    :param sample_pos:
    :return: mX4X4 matrices that used to execute grasp in simulation
    '''

    assert torch.sum(geometry_mask).item() > 0
    depth = depth_projection[geometry_mask]
    gripper_dis_from_source = (0.072 - 0.007 - depth).unsqueeze(dim=-1)
    z_axis = -approaches[geometry_mask,:].reshape(-1,3)  # todo careful
    y_axis = des_normals[geometry_mask,:].reshape(-1,3)
    x_axis = x_axis[geometry_mask,:].reshape(-1,3)


    gripper_position = gripper_dis_from_source.repeat(1, 3) * (-z_axis) + sample_pos[geometry_mask]
    transform_matrix = torch.cat((x_axis.unsqueeze(dim=-1), y_axis.unsqueeze(dim=-1),
                                  z_axis.unsqueeze(dim=-1), gripper_position.unsqueeze(dim=-1)), dim=-1)
    homo_agument = torch.as_tensor([0., 0., 0., 1.]).reshape(1, 1, 4).repeat(len(z_axis), 1, 1).to(des_normals.device)
    transform_matrix = torch.cat((transform_matrix, homo_agument), dim=1)
    return transform_matrix


def get_gripper_points(trans):
    gripper_points_sim = torch.tensor([[0, 0, -0.02, ],
                                       [0.011, -0.09, 0.015, ],
                                       [-0.011, -0.09, 0.015, ],
                                       [0.011, 0.09, 0.015, ],
                                       [-0.011, 0.09, 0.015, ],
                                       [0.008, 0.09, 0.075, ],
                                       [0.008, -0.09, 0.075, ]]).to(torch.float).to(trans.device)

    num_p = gripper_points_sim.size(0)
    gripper_points_sim = gripper_points_sim.unsqueeze(dim=0).repeat(len(trans), 1, 1)
    gripper_points_sim = torch.einsum('pij,pjk->pik', trans[:, :3, :3], gripper_points_sim.transpose(1, 2))
    gripper_points_sim = gripper_points_sim.transpose(1, 2)
    # print(gripper_points_sim.size())
    gripper_points_sim = gripper_points_sim + trans[:, :3, -1].unsqueeze(dim=1).repeat(1, num_p, 1)
    # print(trans[:,:3,-1].unsqueeze(dim=1).repeat(1,5,1))
    return gripper_points_sim

def get_gripper_points_mask(trans, threshold=0.053):
    gripper_points_sim = get_gripper_points(trans)
    z_value = gripper_points_sim[:, :, -1]
    z_mask = z_value > threshold
    z_mask = torch.all(z_mask, dim=1)
    return z_mask

# def save_failure_data(root, grasp_mask, depth_projection, sample_normal, des_normals,
#                       sample_pos, sample, pos, best_score_index,des):
#     scene_id = uuid.uuid4().hex
#     path = root / "failure" / (scene_id + ".npz")
#     np.savez_compressed(path, pos=pos, sample=sample,
#                         depth_projection=depth_projection, sample_normal=sample_normal,
#                         des_normals=des_normals, sample_pos=sample_pos,
#                         grasp_mask=grasp_mask, failure_index=best_score_index,des=des)
#     return scene_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("root", type=Path, default=Path("/data_robot/raw/foo"))
    parser.add_argument("--scene", type=str, choices=["pile", "packed","obj"], default="packed")
    parser.add_argument("--object-set", type=str, default="packed/train")
    parser.add_argument("--sample_number", type=int, default=32)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--sim-gui", action="store_true", default= False)
    parser.add_argument("--baseline", action="store_true", default=True)
    parser.add_argument("--all", action="store_true", default=False)
    parser.add_argument("--add_noise", action="store_true", default=True)
    parser.add_argument("--draw_all", action="store_true", default=False)
    parser.add_argument("--draw_failure", action="store_true", default=False)
    parser.add_argument("--hybrid", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
