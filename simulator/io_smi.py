import json
import uuid

import numpy as np
import pandas as pd
from grasp import Grasp
from perception import *
from transform import Rotation,Transform


def write_setup(root, size, intrinsic, max_opening_width, finger_depth):
    data = {
        "size": size,
        "intrinsic": intrinsic.to_dict(),
        "max_opening_width": max_opening_width,
        "finger_depth": finger_depth,
    }
    write_json(data, root / "setup.json")


def read_setup(root):
    data = read_json(root / "setup.json")
    size = data["size"]
    intrinsic = CameraIntrinsic.from_dict(data["intrinsic"])
    max_opening_width = data["max_opening_width"]
    finger_depth = data["finger_depth"]
    return size, intrinsic, max_opening_width, finger_depth


def write_sensor_data(root, depth_imgs, extrinsics):
    scene_id = uuid.uuid4().hex
    path = root / "scenes" / (scene_id + ".npz")
    np.savez_compressed(path, depth_imgs=depth_imgs, extrinsics=extrinsics)
    return scene_id

def write_mesh_data(root, vertices, faces, vertice_normals):
    scene_id = uuid.uuid4().hex
    path = root / "meshes" / (scene_id + ".npz")
    np.savez_compressed(path, vertices=vertices, faces=faces,vertice_normals=vertice_normals)
    return scene_id

def write_pcd(root, vertices, vertice_normals):
    scene_id = uuid.uuid4().hex
    path = root / "pcd" / (scene_id + ".npz")
    np.savez_compressed(path, vertices=vertices, vertice_normals=vertice_normals)
    return scene_id

def write__mesh_data_with_antipodal(root, vertices, faces, vertice_normals,reasonable_index):
    scene_id = uuid.uuid4().hex
    path = root / "meshes" / (scene_id + ".npz")
    np.savez_compressed(path, vertices=vertices, faces=faces,vertice_normals=vertice_normals,index=reasonable_index)
    return scene_id
def write_mesh_data_with_corr(root, vertices, faces, vertice_normals,reasonable_index,corr):
    scene_id = uuid.uuid4().hex
    path = root / "meshes" / (scene_id + ".npz")
    np.savez_compressed(path, vertices=vertices, faces=faces,vertice_normals=vertice_normals,index=reasonable_index, corr=corr)
    return scene_id

def read_sensor_data(root, scene_id):
    data = np.load(root / "scenes" / (scene_id + ".npz"))
    return data["depth_imgs"], data["extrinsics"]

def read_mesh_data(root, scene_id):
    data = np.load(root / "meshes" / (scene_id + ".npz"))
    return data["vertices"], data["faces"], data['vertice_normals']

def read_mesh_data_antipodal(root, scene_id):
    data = np.load(root / "meshes" / (scene_id + ".npz"))
    return data["vertices"], data["faces"], data['vertice_normals'],data['index']

def write_grasp(root, scene_id, grasp, label):
    # TODO concurrent writes could be an issue
    csv_path = root / "grasps.csv"
    if not csv_path.exists():
        create_csv(
            csv_path,
            ["scene_id", "qx", "qy", "qz", "qw", "x", "y", "z", "width", "label"],
        )
    qx, qy, qz, qw = grasp.pose.rotation.as_quat()
    x, y, z = grasp.pose.translation
    width = grasp.width
    append_csv(csv_path, scene_id, qx, qy, qz, qw, x, y, z, width, label)

def write_grasp_new(root, scene_id, grasp, label,idx_global,antipodal_seen_label,pitch_idx):
    # TODO concurrent writes could be an issue
    csv_path = root / "grasps_new.csv"
    if not csv_path.exists():
        create_csv(
            csv_path,
            ["scene_id", "qx", "qy", "qz", "qw", "x", "y", "z", "width", "label","idx_gobal","antipodal_seen_label","pitch_idx"],
        )
    qx, qy, qz, qw = grasp.pose.rotation.as_quat()
    x, y, z = grasp.pose.translation
    width = grasp.width
    append_csv(csv_path, scene_id, qx, qy, qz, qw, x, y, z, width, label,idx_global,antipodal_seen_label,pitch_idx)

def write_grasp_mutil_labels(root, scene_id, grasp, label, idx_global,pitch_idx,labels):
    # TODO concurrent writes could be an issue
    csv_path = root / "grasps_multi_labels.csv"
    if not csv_path.exists():
        create_csv(
            csv_path,
            ["scene_id", "qx", "qy", "qz", "qw", "x", "y", "z", "width", "label","idx_gobal","pitch_idx",
             "label_0", "label_1", "label_2", "label_3", "label_4", "label_5",
             "label_6", "label_7", "label_8",],
        )
    qx, qy, qz, qw = grasp.pose.rotation.as_quat()
    x, y, z = grasp.pose.translation
    width = grasp.width
    append_csv(csv_path, scene_id, qx, qy, qz, qw, x, y, z, width, label, idx_global, pitch_idx,
               labels[0],labels[1],labels[2],labels[3],labels[4],labels[5],
               labels[6],labels[7],labels[8],)

def write_grasp_corr(root, scene_id, grasp, label,idx_global,idx_reasonable,antipodal_seen_label,pitch_idx):
    # TODO concurrent writes could be an issue
    csv_path = root / "grasps_new.csv"
    if not csv_path.exists():
        create_csv(
            csv_path,
            ["scene_id", "qx", "qy", "qz", "qw", "x", "y", "z", "width", "label","idx_gobal","idx_reasonable","antipodal_seen_label","pitch_idx",],
        )
    qx, qy, qz, qw = grasp.pose.rotation.as_quat()
    x, y, z = grasp.pose.translation
    width = grasp.width
    append_csv(csv_path, scene_id, qx, qy, qz, qw, x, y, z, width, label,idx_global,idx_reasonable,antipodal_seen_label,pitch_idx)

def read_grasp(df, i):
    scene_id = df.loc[i, "scene_id"]
    orientation = Rotation.from_quat(df.loc[i, "qx":"qw"].to_numpy(np.double))
    position = df.loc[i, "x":"z"].to_numpy(np.double)
    width = df.loc[i, "width"]
    label = df.loc[i, "label"]
    grasp = Grasp(Transform(orientation, position), width)
    return scene_id, grasp, label


def read_df(root):
    return pd.read_csv(root / "grasps.csv")


def write_df(df, root):
    df.to_csv(root / "grasps.csv", index=False)


def write_voxel_grid(root, scene_id, voxel_grid):
    path = root / "scenes" / (scene_id + ".npz")
    np.savez_compressed(path, grid=voxel_grid)


def read_voxel_grid(root, scene_id):
    path = root / "scenes" / (scene_id + ".npz")
    return np.load(path)["grid"]


def read_json(path):
    with path.open("r") as f:
        data = json.load(f)
    return data


def write_json(data, path):
    with path.open("w") as f:
        json.dump(data, f, indent=4)


def create_csv(path, columns):
    with path.open("w") as f:
        f.write(",".join(columns))
        f.write("\n")


def append_csv(path, *args):
    row = ",".join([str(arg) for arg in args])
    with path.open("a") as f:
        f.write(row)
        f.write("\n")
