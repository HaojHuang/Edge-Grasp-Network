If you want to generate the data by your own, you need to follow the readme here. Otherwise, go directly to the `model` directory.

## Installation
**Step 1.** Recommended: install `conda` with Python 3.7 (vtk doesn't support 3.8)

```shell
conda create -n grasp_classifier python=3.7
conda activate grasp_classifier
pip install opencv-python pillow scipy matplotlib
conda install mayavi -c conda-forge
```
**Step 2.** Install [Pytorch](https://pytorch.org/get-started/locally/)

**step 3.** Install [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

**Step 4.** Install other required packages

```shell
pip install open3d
pip install pybullet==2.7.9
```

## Getting Started
**Step 1.** Generate training and testing data. Data is saved to the `data_robot/raw/foo` directory.

```shell
python generate_data_grasp_label.py
```
The `pcd` subdirectory contains the `.npz` data with `[vertices=vertices, vertice_normals = vertice_normals]`

**Step 3.** Train a model: go to `model` directory
