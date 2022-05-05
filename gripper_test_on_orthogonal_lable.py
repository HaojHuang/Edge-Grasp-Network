import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import scipy.signal as signal
from grasp import Grasp, Label
from simulation_float_test import ClutterRemovalSim
from transform import Rotation, Transform
from io_smi import *
from utility import FarthestSampler, get_geometry_mask, vis_samples_2, orthognal_grasps,draw_grasps,draw_single_grasp
import warnings
import torch
from torch_geometric.nn import radius
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from data_robot import edge_grasper_revised5
from torch_scatter import scatter
from torch.backends import cudnn
warnings.filterwarnings("ignore")
# OBJECT_COUNT_LAMBDA = 4
# MAX_VIEWPOINT_COUNT = 4


RUN_TIMES = 1
GRASPS_TRIAL_SCENE = 3
NUMBER_SCENE = 200

def main(args):
    device = args.device
    if device == 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')
    grasper = edge_grasper_revised5.EdgeGrasper(device=args.device,root_dir='./data_robot/store16',lr=0.5*1e-4,load=args.load)
    sim = ClutterRemovalSim(args.scene, args.object_set, gui=args.sim_gui)
    finger_depth = sim.gripper.finger_depth
    root = Path('data_robot/raw/foo')
    (root / "pcd").mkdir(parents=True, exist_ok=True)
    (root / "failure").mkdir(parents=True, exist_ok=True)
    write_setup(
        root,
        sim.size,
        sim.camera.intrinsic,
        sim.gripper.max_opening_width,
        sim.gripper.finger_depth, )
    #sim.show_object()

    recoder_level0 = []
    for RUN in range(RUN_TIMES):
        NUMBER_VIEWS = 1
        np.random.seed(RUN+1)
        torch.set_num_threads(RUN+1)
        torch.manual_seed(RUN+1)
        cudnn.benchmark = True
        cudnn.deterministic = True
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
            object_count = 1
            sim.reset(object_count,index=scene_num)
            sim.save_state()
            # render synthetic depth images
            success_object_flag = False
            total_objects += 1
            no_candidate_time = 0
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
                    vertices = vertices + np.random.normal(loc=0.0,scale=0.002, size=(len(vertices),3))
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
                for _ in range(GRASPS_TRIAL_SCENE):
                    sample_number = args.sample_number
                    sample = np.random.choice(len(pos),sample_number,replace=False)
                    #sample = np.unique(sample)
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

                    if sum(orth_mask)>0:
                        #vis_samples_2(pc,sample, radius_p_index[orth_mask].numpy())
                        #vis_samples_2(pc, sample, radius_p_index[angle_mask].numpy())
                        #vis_samples_2(pc, sample, radius_p_index[geometry_mask].numpy())
                        pass

                    if args.baseline:
                        print('Baseline: Random')
                        if sum(geometry_mask)>0:
                            trans_matrix = orthognal_grasps(geometry_mask, depth_projection, sample_normal, des_normals, sample_pos)
                            best_score_index = np.random.randint(low=0, high=sum(geometry_mask))
                            #print(best_score_index)
                            if args.draw_all:
                                draw_grasps(geometry_mask, depth_projection, sample_normal,
                                            normals[radius_p_index, :], sample_pos,
                                            radius_p_index, sample, pos, diverse=False,)

                        else:
                            print('No candidate')
                            no_candidate_time += 1
                            continue

                    else:
                        print('Model: PointNet trained on orthogonal')
                        data = Data(pos=pos.to(torch.float32), normals=normals.to(torch.float32), sample=sample,
                                    radius_p_batch_index=radius_p_batch_index)
                        data.batch = torch.zeros(len(data.pos),dtype=torch.long)
                        # for debug
                        #grasper.test_draw(data, learned=True)
                        data = data.to(device)
                        scores, scores_colli,depth_projection, sample_normal, des_normals, sample_pos, \
                        radius_p_index = grasper.model.act(data)

                        scores = scores.squeeze(dim=-1)
                        scores_colli = scores_colli.squeeze(dim=-1)

                        if args.draw_all:
                            scores_positive_mask = scores>0.5
                            scores_positive = scores[scores_positive_mask].cpu().numpy()
                            score_collision = scores_colli[scores_positive_mask].cpu().numpy()
                            print('score1',scores_positive)
                            print('score2',score_collision)
                            if sum(scores_positive_mask)>1:
                                draw_grasps(scores_positive_mask.cpu(), depth_projection.cpu(), sample_normal.cpu(),
                                            data.normals[radius_p_index, :].cpu(), sample_pos.cpu(),
                                            radius_p_index.cpu(), sample, data.pos.cpu(), diverse=False, scores=score_collision)
                            else: #where the score cannot be normalized
                                draw_grasps(scores_positive_mask.cpu(), depth_projection.cpu(), sample_normal.cpu(),
                                            data.normals[radius_p_index, :].cpu(), sample_pos.cpu(),
                                            radius_p_index.cpu(), sample, data.pos.cpu(), diverse=False,)

                        grasp_mask = torch.logical_and(scores > 0.9, scores_colli > 0.9)
                        if args.hybrid:
                            if sum(geometry_mask)>0:
                                grasp_mask = torch.logical_and(grasp_mask,geometry_mask.to(grasp_mask.device))
                            else:
                                print('No candidate')
                                no_candidate_time += 1
                                continue

                        if sum(grasp_mask)>0:
                            trans_matrix = orthognal_grasps(grasp_mask.cpu(), depth_projection.cpu(), sample_normal.cpu(), des_normals.cpu(),sample_pos.cpu())
                            best_score_index = np.random.randint(low=0, high=sum(grasp_mask.cpu()))
                        else:
                            print('No candidate')
                            no_candidate_time += 1
                            continue

                    if args.all: # try all the positive
                        trans_matrix =trans_matrix.numpy()
                    else:
                        #print(trans_matrix.size())
                        best_trans_matrix = trans_matrix[best_score_index,:,:].unsqueeze(dim=0)
                        #print(best_trans_matrix.size())
                        trans_matrix = best_trans_matrix.numpy()

                    # evaluation
                    res = evaluate_grasps(sim,poses=trans_matrix)
                    success_num, des_list = res
                    success_grasp_totoal += sum(success_num)
                    orthognal_grasp += len(trans_matrix)
                    for idx,des in enumerate(des_list):
                        if des == 'pregrasp':
                            pre +=1
                        if des == 'grasp':
                            grasp_hit +=1
                            if not args.all and not args.baseline:
                                save_failure_data(root, grasp_mask.cpu().numpy(), depth_projection.cpu().numpy(), sample_normal.cpu().numpy(),
                                                  des_normals.cpu().numpy(), sample_pos.cpu().numpy(), sample, pos, best_score_index, des)
                            if args.draw_failure and not args.all:
                                print('collision pose')
                                draw_single_grasp(grasp_mask.cpu(), depth_projection.cpu(), sample_normal.cpu(), des_normals.cpu(),
                                                  sample_pos.cpu(),sample, pos, best_score_index)

                        if des == 'after':
                            after +=1
                            if not args.all and not args.baseline:
                                save_failure_data(root, grasp_mask.cpu().numpy(), depth_projection.cpu().numpy(),
                                                  sample_normal.cpu().numpy(),
                                                  des_normals.cpu().numpy(), sample_pos.cpu().numpy(), sample, pos,
                                                  best_score_index, des)
                            if args.draw_failure and not args.baseline:
                                print('drop pose')
                                draw_single_grasp(grasp_mask.cpu(), depth_projection.cpu(), sample_normal.cpu(),
                                                  des_normals.cpu(),
                                                  sample_pos.cpu(), sample, pos, best_score_index)
                        if des == 'success':
                            des_success_total +=1
                            success_object_flag = True


            if success_object_flag:
                success_object += 1
            if no_candidate_time==3:
                no_candidate_object +=1
            print('Success Object {}, No candidate Object {}, Objects {}, '
                  'Success Grasp {}, Total Grasp {}, '
                  'Pre Collision {}, Failed during Grasping {}, Drop {}, Success_Des {}, '
                  .format(success_object,no_candidate_object, total_objects,
                          success_grasp_totoal,orthognal_grasp,
                          pre,grasp_hit,after,des_success_total,))
            recoder_level1.append([success_object,no_candidate_object, total_objects,
                                   success_grasp_totoal,orthognal_grasp,pre,grasp_hit,after,des_success_total])
            np.save('recoder_level1',np.asarray(recoder_level1))

        recoder_level0.append(recoder_level1)
        np.save('recoder_level0', np.asarray(recoder_level0))


def render_images(sim, n):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0+0.25])
    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)
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

def evaluate_grasps(sim,poses):
    outcomes, widths, describtions = [], [], []
    for i in range(len(poses)):
        pose = poses[i,:,:]
        sim.restore_state()
        candidate = Grasp(Transform.from_matrix(pose), width=sim.gripper.max_opening_width)
        outcome, width, describtion = sim.execute_grasp(candidate, remove=False)
        sim.restore_state()
        outcomes.append(outcome)
        widths.append(width)
        describtions.append(describtion)
        #break
    successes = (np.asarray(outcomes) == Label.SUCCESS).astype(int)
    return successes,describtions


def write_edge_data(root, pos, normals, sample, radius_p_index, radius_p_batch, dot1, dot2, geometry_mask, grasp_label,):
    scene_id = uuid.uuid4().hex
    path = root / "pcd" / (scene_id + ".npz")
    np.savez_compressed(path,pos=pos, normals=normals, sample=sample, radius_p_index=radius_p_index,
                        radius_p_batch=radius_p_batch, dot1=dot1, dot2=dot2, geometry_mask=geometry_mask, grasp_label=grasp_label)
    return scene_id


def save_failure_data(root, grasp_mask, depth_projection, sample_normal, des_normals,
                      sample_pos, sample, pos, best_score_index,des):
    scene_id = uuid.uuid4().hex
    path = root / "failure" / (scene_id + ".npz")
    np.savez_compressed(path, pos=pos, sample=sample,
                        depth_projection=depth_projection, sample_normal=sample_normal,
                        des_normals=des_normals, sample_pos=sample_pos,
                        grasp_mask=grasp_mask, failure_index=best_score_index,des=des)
    return scene_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("root", type=Path, default=Path("/data_robot/raw/foo"))
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="pile")
    parser.add_argument("--object-set", type=str, default="pile/train")
    parser.add_argument("--load", type=int, default=190)
    parser.add_argument("--sample_number", type=int, default=32)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--sim-gui", action="store_true", default= True)
    parser.add_argument("--baseline", action="store_true", default=True)
    parser.add_argument("--all", action="store_true", default=False)
    parser.add_argument("--add_noise", action="store_true", default=False)
    parser.add_argument("--draw_all", action="store_true", default=False)
    parser.add_argument("--draw_failure", action="store_true", default=False)
    parser.add_argument("--hybrid", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
