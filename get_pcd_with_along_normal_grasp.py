import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import scipy.signal as signal
from grasp import Grasp,Label
from perception import *
from simulation_float import ClutterRemovalSim
from transform import Rotation,Transform
from io_smi import *
from utility import FarthestSampler
import warnings

warnings.filterwarnings("ignore")
#OBJECT_COUNT_LAMBDA = 4
#MAX_VIEWPOINT_COUNT = 4

GRASPS_PER_SCENE = 200
NUMBER_SCENE = 80
NUMBER_VIEWS = 5

def main(args):

    sim = ClutterRemovalSim(args.scene,args.object_set, gui=args.sim_gui)
    finger_depth = sim.gripper.finger_depth
    (args.root / "pcd").mkdir(parents=True,exist_ok=True)
    write_setup(
        args.root,
        sim.size,
        sim.camera.intrinsic,
        sim.gripper.max_opening_width,
        sim.gripper.finger_depth,
    )
    success_num = 0
    first = 0
    pre = 0
    after= 0
    grasp_hit = 0
    grasp_success = 0

    for scene_num in range(NUMBER_SCENE):
        object_count = 1
        sim.reset(object_count)
        sim.save_state()
        # render synthetic depth images
        for partial_view in range(NUMBER_VIEWS):
            n=1
            depth_imgs, extrinsics, eye = render_images(sim, n)
            # reconstrct point cloud using a subset of the images
            tsdf = create_tsdf(sim.size, 180, depth_imgs, sim.camera.intrinsic, extrinsics)
            pc = tsdf.get_cloud()
            camera_location = eye
            # crop surface and borders from point cloud
            bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
            pc = pc.crop(bounding_box)
            #o3d.visualization.draw_geometries([pc])
            if pc.is_empty():
                print("Empty point cloud, skipping scene")
                continue

            pc, ind = pc.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
            pc, ind = pc.remove_radius_outlier(nb_points=30, radius=0.05)
            pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pc.orient_normals_consistent_tangent_plane(30)
            pc.orient_normals_towards_camera_location(camera_location=camera_location)
            normals = np.asarray(pc.normals)
            # direction = -np.asarray(pc.points) + camera_location
            # dot = np.sum(normals * direction, axis=-1)
            # mask = dot < -0.0
            # normals[mask] *= -1
            vertices = np.asarray(pc.points)
            if len(vertices)<200:
                continue
            if len(vertices)>1024:
                sampler = FarthestSampler()
                vertices, idx = sampler(vertices, 1024)
                normals = normals[idx]
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(vertices)
                pc.normals = o3d.utility.Vector3dVector(normals)

            #o3d.visualization.draw_geometries([pc], point_show_normal=True)
            #print(len(vertices))
            # store the raw data
            scene_id = write_pcd(args.root,vertices,normals)
            # sample grasp points
            sampler = FarthestSampler()
            _, sample_index_list_random = sampler(vertices,GRASPS_PER_SCENE)
            #display_inlier_outlier(pc,sample_index_list_random)
            eps = 0.1
            for i in range(GRASPS_PER_SCENE):
                idx_global = sample_index_list_random[i]
                #display_inlier_outlier(pc, [idx_global])
                p, n, l = vertices[idx_global], normals[idx_global], sim.gripper.max_opening_width
                #grasp_depth = np.random.uniform(4 * eps * finger_depth, (1.0 - 2*eps) * finger_depth)
                grasp_depth = 0.045
                p = p + n * grasp_depth
                res = evaluate_grasp_point_3(sim,p,n,l)
                if res is None:
                    continue
                else:
                    grasp, label, des_list, idx_of_peak, labels = res

                write_grasp_mutil_labels(args.root, scene_id, grasp, label, idx_global, idx_of_peak,labels)
                #print('label', label)

                if label ==1:
                    success_num += 1

                if des_list[0] == 'success':
                    first += 1
                for des in des_list:
                    if des == 'pregrasp':
                        pre +=1
                    if des == 'grasp':
                        grasp_hit +=1
                    if des == 'after':
                        after +=1
                    if des == 'success':
                        grasp_success += 1
            #print('===================')

        num_trials = GRASPS_PER_SCENE*9*(NUMBER_VIEWS)*(scene_num+1)
        print('Scene[{}/{}], First Success: {}/{}, Pregrasp Collision: {}/{}, Grasp Collision: {}/{},'
              ' Slipping after grasp: {}/{}, Success Trial: {}/{}, Success Points: {}/{}'.format(scene_num+1,NUMBER_SCENE,first,
                                                                                                 num_trials,pre,num_trials,grasp_hit,num_trials,
                                                                                                after,num_trials,grasp_success,num_trials,success_num,num_trials/9))



def render_images(sim, n):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.25])

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)

    for i in range(n):
        r = np.random.uniform(1.6, 2.4) * sim.size
        theta = np.random.uniform(0.0, np.pi)
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


    return depth_imgs, extrinsics,eye


def normalization(x):
    x_norm = max(np.linalg.norm(x), 1e-12)
    x = x / x_norm
    return x


def evaluate_grasp_point_3(sim,point,normal,length,num_rotations=9):
    # Todo by haojie: need to explore the whether the classfication makes sense
    z_axis = -normal
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-3):
        return
        #x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)
    yaws = np.linspace(0.0, np.pi, num_rotations,endpoint=False)
    #print(yaws)
    outcomes, widths,describtions = [], [], []
    for yaw in yaws:
        ori = R*Rotation.from_euler('z', yaw)
        sim.restore_state()
        candidate = Grasp(Transform(ori, point), width=sim.gripper.max_opening_width)
        outcome, width, describtion = sim.execute_grasp(candidate, remove=False)
        sim.restore_state()
        outcomes.append(outcome)
        widths.append(width)
        describtions.append(describtion)
    # TODO currently this does not properly handle periodicity
    successes = (np.asarray(outcomes) == Label.SUCCESS).astype(float)
    idx_of_widest_peak = -1
    if np.sum(successes):
        peaks, properties = signal.find_peaks(
            x=np.r_[0, successes, 0], height=1, width=1
        )
        idx_of_widest_peak = peaks[np.argmax(properties["widths"])] - 1
        ori = R * Rotation.from_euler("z", yaws[idx_of_widest_peak])
        width = widths[idx_of_widest_peak]

    return Grasp(Transform(ori, point), width), int(np.max(outcomes)), describtions, idx_of_widest_peak, successes

def display_inlier_outlier(cloud, ind, show_total=True):

    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    print(cloud)
    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    inlier_cloud.paint_uniform_color([1, 0, 0])
    if show_total:
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],)
    else:
        o3d.visualization.draw_geometries([inlier_cloud])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="pile")
    parser.add_argument("--object-set", type=str, default="pile/train")
    parser.add_argument("--num-grasps", type=int, default=10000)
    parser.add_argument("--sim-gui", action="store_true",default=True)
    args = parser.parse_args()
    main(args)
