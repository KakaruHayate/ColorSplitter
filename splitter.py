import pandas as pd
from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from modules.Resemblyzer.visualizations import *
from modules.speakerlab.cluster import CommonClustering  
import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--spk', type=str, help='Speaker name')
parser.add_argument('--nmin', type=int, default=1, help='minimum number of clusters')
parser.add_argument('--cluster', type=int, default=1, help='1:SpectralCluster, 2:UmapHdbscan')
parser.add_argument('--mer_cosine', type=str, default=None, help='merge similar timbre')
args = parser.parse_args()

Speaker_name = args.spk #Speaker name
Nmin = args.nmin # set Nmax values
merge_cos = args.mer_cosine

data_dir = os.path.join("input", Speaker_name, "raw", "wavs")
wav_fpaths = list(Path(data_dir).glob("*.wav"))

encoder = VoiceEncoder(weights_fpath="pretrain/encoder_1570000.bak")

# Check if features already exist
features_path = os.path.join("input", Speaker_name, "features(timbre).pkl")
if os.path.exists(features_path):
    with open(features_path, 'rb') as f:
        resemblyzer_embeds = pickle.load(f)
else:
    wavs = [preprocess_wav(wav_fpath) for wav_fpath in \
            tqdm(wav_fpaths, f"Preprocessing wavs ({len(wav_fpaths)} utterances)")] 
    resemblyzer_embeds = np.array(list(map(encoder.embed_utterance, wavs)))
    with open(features_path, 'wb') as f:
        pickle.dump(resemblyzer_embeds, f)

while True:
    if args.cluster == 1:
        cluster_name = 'spectral'
        min_num_spks=Nmin
        mer_cos=merge_cos
        Cluster = CommonClustering(cluster_type=cluster_name, mer_cos=None, min_num_spks=Nmin)
    elif args.cluster == 2:
        cluster_name = 'umap_hdbscan'
        mer_cos=merge_cos
        Cluster = CommonClustering(mer_cos=None, cluster_type=cluster_name)
    else:
        raise ValueError('cluster type error')

    labels = Cluster.__call__(resemblyzer_embeds)

    output_dir = f'output/{Speaker_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.DataFrame({
        'filename': [str(fpath) for fpath in wav_fpaths],
        'clust': labels
    })
    df.to_csv(f'{output_dir}/clustered_files(timbre).csv', index=False)

    plot_projections(resemblyzer_embeds, labels, title="Embedding projections", cluster=cluster_name)
    plt.savefig(f'{output_dir}/embedding_projections(timbre).png', dpi=600)
    plt.show()

    user_input = input("Are you satisfied with the results?/是否满意结果？(y/n): ")
    if user_input.lower() == 'y':
        break
    else:
        Nmin = int(input("Please enter a new Nmin value/请输入新的Nmin值: "))
