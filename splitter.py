import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from modules.utils import GetEmbeds
from modules.visualizations import plot_projections
from modules.cluster import CommonClustering
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--spk", type=str, help="Speaker name")
parser.add_argument("--nmin", type=int, default=1, help="minimum number of clusters")
parser.add_argument(
    "--cluster", type=int, default=1, help="1:SpectralCluster, 2:UmapHdbscan"
)
parser.add_argument("--mer_cosine", type=str, default=None, help="merge similar embeds")
parser.add_argument("--encoder", type=str, default="timbre", help="encoder type")
args = parser.parse_args()

Speaker_name = args.spk  # Speaker name
Nmin = args.nmin  # set Nmax values
merge_cos = args.mer_cosine
encoder_name = args.encoder

data_dir = os.path.join("input", Speaker_name, "raw", "wavs")
wav_fpaths = list(Path(data_dir).glob("*.wav"))

encoder = GetEmbeds(encoder_type=encoder_name, Speaker_name=Speaker_name)

embeds = encoder.__call__(wav_fpaths)

while True:
    if args.cluster == 1:
        cluster_name = "spectral"
        min_num_spks = Nmin
        mer_cos = merge_cos
        Cluster = CommonClustering(
            cluster_type=cluster_name, mer_cos=None, min_num_spks=Nmin
        )
    elif args.cluster == 2:
        cluster_name = "umap_hdbscan"
        mer_cos = merge_cos
        Cluster = CommonClustering(mer_cos=None, cluster_type=cluster_name)
    else:
        raise ValueError("cluster type error")

    labels = Cluster.__call__(embeds)

    output_dir = f"output/{Speaker_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.DataFrame(
        {"filename": [str(fpath) for fpath in wav_fpaths], "clust": labels}
    )
    proj = plot_projections(
        embeds, labels, title="Embedding projections", cluster_name=cluster_name
    )
    df["x"] = proj[:, 0]
    df["y"] = proj[:, 1]

    plt.savefig(f"{output_dir}/embedding_projections({encoder_name}).png", dpi=600)
    plt.show()
    df.to_csv(f"{output_dir}/clustered_files({encoder_name}).csv", index=False)

    user_input = input("Are you satisfied with the results?/是否满意结果？(y/n): ")
    if user_input.lower() == "y":
        break
    else:
        Nmin = int(input("Please enter a new Nmin value/请输入新的Nmin值: "))
