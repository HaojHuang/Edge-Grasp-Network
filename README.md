# Panda-gripper Simulation Environment for Grasp

## Installation
**Step 1.** Recommended: install `conda` with Python 3.8.

```shell
conda create -n grasp_classifier python=3.8
```

**Step 2.** Install Pytorch


**Step 3.** Install the required packages

```shell
pip install open3d
pip install pybullet==2.7.9
```

## Getting Started
**Step 1.** Generate training and testing data. Data is saved to the `data_robot/raw/foo` directory.

The `pcd` subdirectory contains the `.npz` data with `[vertices=vertices, vertice_normals = vertice_normals]`, the 
`grasp_multi_labels.csv` contains the detailed grasp information of each sampled point.

Or download generated data [here](https://github.com/HaojHuang/grasp_classifier/blob/main/raw.zip).

```shell
python get_pcd_with_along_normal_grasp.py data_robot/raw/foo
```

**Step 2.** Create a dataset with [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) **To do**

**Step 3.** Train a model **To do**
