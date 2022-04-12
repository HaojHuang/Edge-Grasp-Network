
**Step 1.** unzip the raw.zip file (raw1.zip is the most recent one)

**Step 2.** try the vis_data.py to explore the data structure

```commandline
python vis_data.py
```


**Step 3.** install [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

**Step 4.** Generate the PyG-style Dataset

```
python edge_grasper_dataset.py
```

**Step 5.** Visual the grasp pose 

```
python vis_grasp.py
```

**Step 6.** Train the model: The classifier is composed of a MLPs with sigmoid as the last activation layer.

```
python edge_grasper.py
```
