import argparse
import time
from pathlib import Path
import numpy
import numpy as np
import open3d as o3d
from simulator.grasp import Grasp, Label
from simulator.simulation_clutter_bandit import ClutterRemovalSim
from simulator.transform import Rotation, Transform
from simulator.io_smi import *
from simulator.utility import FarthestSamplerTorch, get_gripper_points_mask, orthognal_grasps, FarthestSampler
#from utility import bandit_grasp
import warnings
import torch
from torch_geometric.nn import radius
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch.backends import cudnn
from termcolor import colored
from models.edge_grasper import EdgeGrasper
from models.vn_edge_grasper import EdgeGrasper as VNGrasper
import tqdm
warnings.filterwarnings("ignore")
# OBJECT_COUNT_LAMBDA = 4
# MAX_VIEWPOINT_COUNT = 4

RUN_TIMES = 4
NUMBER_SCENE = 100
NUMBER_VIEWS = 1
OBJECT_COUNT_LAMBDA = 4
GRASP_PER_SCENE = 1

def main(args):
    device = args.device
    if device == 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')
    if args.vn:
        grasper = VNGrasper(device=args.device, root_dir='./vn_edge_pretrained_para', load=105)
    else:
        grasper = EdgeGrasper(device=args.device, root_dir='./edge_grasp_net_pretrained_para', load=180)

    sim = ClutterRemovalSim(args.scene, args.object_set, gui=args.sim_gui,rand=args.rand)
    # finger_depth = sim.gripper.finger_depth
    # root = Path('data_robot/raw/foo_implicit')
    # (root / "pcd").mkdir(parents=True, exist_ok=True)
    # write_setup(
    #     root,
    #     sim.size,
    #     sim.camera.intrinsic,
    #     sim.gripper.max_opening_width,
    #     sim.gripper.finger_depth, )
    # # sim.show_object()
    # memory = ReplayMemory(capacity=1000)
    record = []
    for RUN in range(RUN_TIMES):
        np.random.seed(RUN + 1)
        torch.set_num_threads(RUN + 1)
        torch.manual_seed(RUN + 1)
        cudnn.benchmark = True
        cudnn.deterministic = True
        num_rounds = 100
        silence = False
        cnt = 0
        success = 0
        left_objs = 0
        total_objs = 0
        cons_fail = 0
        for _ in tqdm.tqdm(range(num_rounds), disable=silence):
            object_count = np.random.poisson(OBJECT_COUNT_LAMBDA) + 1
            object_count = 5
            sim.reset(object_count)
            ###
            total_objs += sim.num_objects
            consecutive_failures = 1
            last_label = None
            trial_id = -1
            ###
            skip_time = 0
            empyty = False
            while sim.num_objects>0 and consecutive_failures<2 and skip_time<3:
                trial_id += 1
                # use one camera views
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
                    empyty = True
                    break
                if args.add_noise:
                    vertices = np.asarray(pc.points)
                    # add gaussian noise 95% confident interval (-1.96,1.96)
                    vertices = vertices + np.random.normal(loc=0.0, scale=0.0008, size=(len(vertices), 3))
                    pc.points = o3d.utility.Vector3dVector(vertices)

                vertices = np.asarray(pc.points)
                if len(vertices) < 100:
                    print("point cloud<100, skipping scene")
                    #skip_time += 1
                    empyty=True
                    break
                time0 = time.time()
                pc, ind = pc.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
                pc, ind = pc.remove_radius_outlier(nb_points=30, radius=0.03)
                pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
                pc.orient_normals_consistent_tangent_plane(30)
                # orient the normals direction
                #pc.orient_normals_towards_camera_location(camera_location=camera_location)
                normals = np.asarray(pc.normals)
                vertices = np.asarray(pc.points)
                if len(vertices) < 100:
                    print("point cloud<100, skipping scene")
                    skip_time += 1
                    break

                pc = pc.voxel_down_sample(voxel_size=0.0045)
                # continue
                #o3d.visualization.draw_geometries([pc], point_show_normal=True)
                # break
                #########################
                pos = np.asarray(pc.points)
                #print(len(pos))
                normals = np.asarray(pc.normals)
                pos = torch.from_numpy(pos).to(torch.float32).to(device)
                # print('min z, max z', pos[:,-1].min(), pos[:,-1].max())
                normals = torch.from_numpy(normals).to(torch.float32).to(device)
                sample_number = args.sample_number

                fps_sample = FarthestSamplerTorch()
                _, sample = fps_sample(pos,sample_number)
                sample = torch.as_tensor(sample).to(torch.long).reshape(-1).to(device)
                sample = torch.unique(sample,sorted=True)
                #print(sample)
                #sample = np.random.choice(len(pos), sample_number, replace=False)
                #sample = torch.from_numpy(sample).to(torch.long)
                sample_pos = pos[sample, :]
                radius_p_batch_index = radius(pos, sample_pos, r=0.05, max_num_neighbors=1024)
                radius_p_index = radius_p_batch_index[1, :]
                radius_p_batch = radius_p_batch_index[0, :]
                sample_pos = torch.cat(
                    [sample_pos[i, :].repeat((radius_p_batch == i).sum(), 1) for i in range(len(sample))],
                    dim=0)
                sample_copy = sample.clone().unsqueeze(dim=-1)
                sample_index = torch.cat(
                    [sample_copy[i, :].repeat((radius_p_batch == i).sum(), 1) for i in range(len(sample))], dim=0)
                edges = torch.cat((sample_index, radius_p_index.unsqueeze(dim=-1)), dim=1)
                #all_edge_index = numpy.arange(0, len(edges))
                all_edge_index = torch.arange(0,len(edges)).to(device)
                des_pos = pos[radius_p_index, :]
                des_normals = normals[radius_p_index, :]
                relative_pos = des_pos - sample_pos


                # fps_torch = FarthestSamplerTorch()
                # edge_global_idx = []
                # for i in range(args.sample_number):
                #     batch_mask = radius_p_batch == i
                #     num = torch.sum(batch_mask)
                #
                #     if num < 70:
                #         edge_global_idx.append(all_edge_index[batch_mask])
                #     else:
                #         des_batch_pos = des_pos[batch_mask, :]
                #         des_batch_index = all_edge_index[batch_mask]
                #         _, idx = fps_torch(des_batch_pos, 60)
                #         edge_global_idx.append(des_batch_index[idx])
                # edge_global_idx = torch.cat(edge_global_idx, dim=0)
                # # print(edge_global_idx.size())

                relative_pos_normalized = F.normalize(relative_pos, p=2, dim=1)
                # set up the record
                label_record = []
                # edge_sample_index = []
                quat_record = []
                translation_record = []
                # only record approach vectors with a angle mask
                x_axis = torch.cross(des_normals, relative_pos_normalized)
                x_axis = F.normalize(x_axis, p=2, dim=1)
                valid_edge_approach = torch.cross(x_axis, des_normals)
                valid_edge_approach = F.normalize(valid_edge_approach, p=2, dim=1)
                valid_edge_approach = -valid_edge_approach
                # print('new approachs',valid_edge_approach.shape)
                up_dot_mask = torch.einsum('ik,k->i', valid_edge_approach, torch.tensor([0., 0., 1.]).to(device))
                relative_norm = torch.linalg.norm(relative_pos, dim=-1)
                # print(relative_norm.size())
                depth_proj = -torch.sum(relative_pos * valid_edge_approach, dim=-1)
                geometry_mask = torch.logical_and(up_dot_mask > -0.1, relative_norm > 0.003)
                geometry_mask = torch.logical_and(relative_norm<0.038,geometry_mask)
                depth_proj_mask = torch.logical_and(depth_proj > -0.000, depth_proj < 0.04)
                geometry_mask = torch.logical_and(geometry_mask, depth_proj_mask)
                if torch.sum(geometry_mask)<10:
                    skip_time+=1
                    continue
                # draw_grasps2(geometry_mask, depth_proj, valid_edge_approach, des_normals, sample_pos, pos, sample, des=None, scores=None)
                pose_candidates = orthognal_grasps(geometry_mask, depth_proj, valid_edge_approach, des_normals,
                                                   sample_pos)
                table_grasp_mask = get_gripper_points_mask(pose_candidates,threshold=0.054)
                # print('no collision with table candidates all', table_grasp_mask.sum())
                geometry_mask[geometry_mask == True] = table_grasp_mask
                # wether fps
                # geometry_mask = torch.logical_and(geometry_mask,geometry_mask2)
                edge_sample_index = all_edge_index[geometry_mask]
                # print('no collision with table candidates', len(edge_sample_index))
                if len(edge_sample_index) > 0:
                    if len(edge_sample_index) > 1500:
                        edge_sample_index = edge_sample_index[torch.randperm(len(edge_sample_index))[:1500]]
                    edge_sample_index, _ = torch.sort(edge_sample_index)
                    # print('candidate numbers', len(edge_sample_index))
                    data = Data(pos=pos, normals=normals, sample=sample, radius_p_index=radius_p_index,
                                ball_batch=radius_p_batch,
                                ball_edges=edges, approaches=valid_edge_approach[edge_sample_index, :],
                                reindexes=edge_sample_index,
                                relative_pos=relative_pos[edge_sample_index, :],
                                depth_proj=depth_proj[edge_sample_index])

                    data = data.to(device)
                    score, depth_projection, approaches, sample_pos, des_normals = grasper.model.act(data)
                    if not args.baseline:
                        # max_indice = torch.argmax(score)
                        # print(score.size())
                        k_score, max_indice = torch.topk(score, k=1)
                        selected_edge = edges[edge_sample_index[max_indice],:]
                        max_score = score[max_indice]
                        max_score = F.sigmoid(max_score).cpu().numpy()
                        #print('max score', max_score)
                        if max_score.any() < 0.85:
                            print('no confident on this observation, skip')
                            skip_time+=1
                            continue
                    else:
                        print('baseline')
                        max_indice = torch.randint(low=0, high=len(score),size=(1,))

                    grasp_mask = torch.ones(len(depth_projection)) > 2.
                    grasp_mask[max_indice] = True
                    trans_matrix = orthognal_grasps(grasp_mask.to(des_normals.device), depth_projection, approaches,
                                                    des_normals, sample_pos)
                    trans_matrix = trans_matrix.cpu().numpy()
                    if args.point_sample:
                        print('point sample')
                        trans_matrix = sample_grasp_point(pc)
                    ## width
                    if args.width:
                        widthes = torch.abs(torch.sum(data.relative_pos * des_normals, dim=-1)) + 0.016
                        widthes = widthes[grasp_mask].clip(max=0.04)
                        widthes = (widthes * 2).cpu().numpy()
                    else:
                        widthes = None
                    # evaluation
                    res = evaluate_grasps(sim, poses=trans_matrix,width_pre=widthes)
                    success_num, des_list, quats, translations = res
                    cnt += 1
                    label = success_num[0]
                    if label!=Label.FAILURE:
                        success+=1
                    if last_label == Label.FAILURE and label == Label.FAILURE:
                        consecutive_failures += 1
                    else:
                        consecutive_failures = 1

                    if consecutive_failures >= 2:
                        cons_fail += 1
                    last_label = label
                else:
                    print('no candidates without collision')
                    skip_time+=1
                    continue

                left_objs += sim.num_objects

            success_rate = 100.0 * success / cnt
            declutter_rate = 100.0 * success / total_objs
            print('success grasp:' ,success, 'total grasp:', cnt, 'total objects:', total_objs)
            print('Grasp success rate: %.2f %%, Declutter rate: %.2f %%' % (success_rate, declutter_rate))

        log = [success_rate, declutter_rate]
        record.append(log)
        #print(record)
        scene_name = str(args.scene)
        sample_num = args.sample_number
        np.save('clutter_record_{}_{}'.format(scene_name,sample_num), np.asarray(record))

def render_images(sim, n):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0 + 0.25])
    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)
    for i in range(n):
        r = np.random.uniform(2, 2.5) * sim.size
        theta = np.random.uniform(np.pi/4, np.pi/3)
        phi = np.random.uniform(0.0, np.pi)

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

def sample_grasp_point(point_cloud, finger_depth=0.05, eps=0.1):
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    ok = False
    while not ok:
        # TODO this could result in an infinite loop, though very unlikely
        idx = np.random.randint(len(points))
        point, normal = points[idx], normals[idx]
        ok = normal[2] > -0.1  # make sure the normal is poitning upwards
    grasp_depth = np.random.uniform(-eps * finger_depth, (1.0 + eps) * finger_depth)
    point = point + normal * grasp_depth # match the tcp point
    z_axis = -normal
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)
    # try to grasp with a random yaw angles
    yaws = np.linspace(0.0, np.pi, 12, endpoint=False)
    idx = np.random.randint(len(yaws))
    yaw = yaws[idx]
    ori = R * Rotation.from_euler("z", yaw)
    pose = Transform(ori, point).as_matrix()[np.newaxis,...]
    return pose

def evaluate_grasps(sim, poses, width_pre=None):
    outcomes, widths, describtions = [], [], []
    quats, translations = [], []
    for i in range(len(poses)):
        pose = poses[i, :, :]
        #sim.restore_state()
        dof_6 = Transform.from_matrix(pose)
        # decompose the quat
        quat = dof_6.rotation.as_quat()
        translation = dof_6.translation
        if width_pre is not None:
            width = width_pre[i]
        else:
            width = sim.gripper.max_opening_width

        candidate = Grasp(dof_6, width=width)
        # outcome, width, describtion = sim.execute_grasp(candidate, remove=False)
        outcome, width, describtion = sim.execute_grasp_quick(candidate,allow_contact=True,remove=True)
        #sim.restore_state()
        outcomes.append(outcome)
        widths.append(width)
        describtions.append(describtion)
        quats.append(quat)
        translations.append(translation)
        # break
    successes = (np.asarray(outcomes) == Label.SUCCESS).astype(int)
    return successes, describtions, quats, translations



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("root", type=Path, default=Path("/data_robot/raw/foo"))
    #choices = ["pile", "packed", "obj", "egad"]
    parser.add_argument("--scene", type=str, choices=["pile", "packed", "obj", "egad"], default="packed")
    parser.add_argument("--object-set", type=str, default="packed/test")
    parser.add_argument("--sample_number", type=int, default=32)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--grasp_obs", type=int, default=1)
    parser.add_argument("--sim-gui", action="store_true", default =True)
    parser.add_argument("--rand", action="store_true", default = True)
    parser.add_argument("--width", action="store_true", default=False)
    parser.add_argument("--baseline", action="store_true", default=False)
    parser.add_argument("--point_sample", action="store_true", default=False)
    parser.add_argument("--all", action="store_true", default=False)
    parser.add_argument("--add_noise", action="store_true", default=True)
    parser.add_argument("--draw_all", action="store_true", default=False)
    parser.add_argument("--draw_failure", action="store_true", default=False)
    parser.add_argument("--hybrid", action="store_true", default=False)
    parser.add_argument("--vn", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
