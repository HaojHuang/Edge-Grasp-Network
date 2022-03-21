import pandas as pd
from pathlib import Path
import numpy as np

def write_df(df, root):
    df.to_csv(root / "grasps_clean.csv", index=False)


root = Path('c_mesh_normal_test')
df = pd.read_csv(root / "grasps_new.csv")
#print(df)
# remove grasp positions that lie outside the workspace
# df.drop(df[df["x"] < 0.01].index, inplace=True)
# df.drop(df[df["y"] < 0.01].index, inplace=True)
# df.drop(df[df["z"] < 0.01].index, inplace=True)
# df.drop(df[df["x"] > 0.28].index, inplace=True)
# df.drop(df[df["y"] > 0.28].index, inplace=True)
# df.drop(df[df["z"] > 0.28].index, inplace=True)


scenes = df["scene_id"].values
#print(scenes,len(scenes))
for f in (root / "meshes").iterdir():
    #print(str(f.stem))
    if f.suffix == ".npz" and f.stem not in scenes:
        print("Removed", f)
        f.unlink()

scene_root = root / "meshes"
scene_ids = [f for f in scene_root.iterdir() if f.suffix == ".npz"]
print(len(scene_ids))


for i in range(len(scene_ids)):
    scene_id = str(scene_ids[i])
    sparsed = scene_id.split('/')
    #print(sparsed)
    scene_id = sparsed[-1][:-4]
    #print(scene_id)
    scene_df = df[df["scene_id"]==scene_id]
    scene_df_positive = scene_df[scene_df["label"] == 1]
    scene_df_negative = scene_df[scene_df["label"] == 0]
    #print(len(scene_df_positive),len(scene_df_negative))
    if len(scene_df_positive.index)==0:
        df.drop(scene_df.index,inplace=True)
        continue
    # if len(scene_df_positive.index) > len(scene_df_negative.index):
    #     #print(scene_id,'more than 50%',len(scene_df_positive.index),len(scene_df_negative.index))
    #     j = np.random.choice(scene_df_positive.index, len(scene_df_positive.index) - len(scene_df_negative.index), replace=False)
    #     df = df.drop(j)
    # if len(scene_df_positive.index) < len(scene_df_negative.index):
    #     j = np.random.choice(scene_df_negative.index, len(scene_df_negative.index) - len(scene_df_positive.index), replace=False)
    #     df = df.drop(j)


positives = df[df["label"] == 1]
negatives = df[df["label"] == 0]
print("Number of samples:", len(df.index))
print("Number of positives:", len(positives.index))
print("Number of negatives:", len(negatives.index))

write_df(df,root)
