import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from modules.demo_utils import *
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--spk', type=str, help='Speaker name')
parser.add_argument('--nmax', type=int, default=10, help='Maximum number of clusters')
args = parser.parse_args()

Speaker_name = args.spk #Speaker name
Nmax = args.nmax # set Nmax values

data_dir = os.path.join("input", Speaker_name, "raw", "wavs")
wav_fpaths = list(Path(data_dir).glob("*.wav"))

encoder = VoiceEncoder(weights_fpath="pretrain/encoder_112500.bak")

wavs = [preprocess_wav(wav_fpath) for wav_fpath in \
        tqdm(wav_fpaths, f"Preprocessing wavs ({len(wav_fpaths)} utterances)")]
        
resemblyzer_embeds = np.array(list(map(encoder.embed_utterance, wavs)))
utterance_embeds = np.nan_to_num(resemblyzer_embeds)

Cluster_per_N = []
for i in tqdm(range(2, Nmax + 1), desc = "Clustering..."):
    Cluster = SpectralClustering(n_clusters=i, affinity='nearest_neighbors', random_state=0, gamma=1.0, n_init = 10).fit(utterance_embeds)
    
    plot_projections(utterance_embeds, Cluster.labels_, title="Embedding projections")
    
    output_dir = f'output/{Speaker_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/embedding_projections_{i}.png', dpi=600)

    df = pd.DataFrame({
        'filename': [str(fpath) for fpath in wav_fpaths],
        'clust': Cluster.labels_
    })

    df.to_csv(f'{output_dir}/clustered_files_{i}.csv', index=False)
    Cluster_per_N.append(Cluster)
     
scores = [metrics.silhouette_score(utterance_embeds, model.labels_) for model in tqdm(Cluster_per_N, desc = "Programming scores")]

plt.figure(figsize = (8, 4))
plt.plot(range(2, Nmax + 1), scores, 'bo-')
plt.savefig(f'{output_dir}/scores.png', dpi=600)
plt.show()