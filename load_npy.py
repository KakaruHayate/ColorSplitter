import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from modules.visualizations import plot_projections, process_json_file
from modules.cluster import CommonClustering
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the .npy file')
parser.add_argument('--reducer', type=int, default=2, help='1:tSNE, 2:Umap')
parser.add_argument('--json', type=str, default=None, help='path to the .json file')
args = parser.parse_args()

if args.reducer == 1:
	cluster_name = 'spectral'
elif args.reducer == 2:
	cluster_name = 'umap_hdbscan'
else:
	raise ValueError('reducer type error')

npy_path = args.path

embeds = np.load(npy_path)

if args.json == None:
	token_names = np.arange(embeds.shape[0])
else:
	token_names = process_json_file(args.json)
labels = np.ones_like(token_names)

output_dir = f'output/npy_result'
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

df = pd.DataFrame({
	'token': [f'{i}' for i in range(embeds.shape[0])],
	'clust': labels
})
df.to_csv(f'{output_dir}/clustered_files({os.path.basename(npy_path)}).csv', index=False)


plot_projections(embeds, labels, title="Embedding projections", cluster_name=cluster_name, labels=token_names)
plt.savefig(f'{output_dir}/embedding_projections({os.path.basename(npy_path)}).png', dpi=600)
plt.show()
