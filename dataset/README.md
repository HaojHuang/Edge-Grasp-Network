
**Step 1.** unzip the raw.zip file

**Step 2.** try the vis_data.py to explore the data structure

```commandline
python vis_data.py
```


**Step 3.** install [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

**Step 4.** Generate the PyG-style Dataset

```
python dataset_pyg.py
```

**Step 5.** Visual the grasp pose 

```
python vis_grasp.py
```

**Step 6.** Train the model: aggregation.py uses PointNetConv; aggregation_2 uses PPFConv. The classifier is composed of a shallow MLPs with sigmoid as the last layer.

```
python train.py
```
