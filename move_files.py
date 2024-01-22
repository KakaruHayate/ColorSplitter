import os
import shutil
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--spk', type=str, help='Speaker name')
parser.add_argument('--encoder', type=str, default='timbre', help='encoder type')
args = parser.parse_args()


Speaker_name = args.spk #Speaker name
encoder_name = args.encoder
 
data = pd.read_csv(os.path.join('output', Speaker_name, f'clustered_files({encoder_name}).csv'))

for index, row in data.iterrows():
    file_path = row['filename']
    clust = row['clust']

    clust_dir = os.path.join('output', Speaker_name, str(clust))
    if not os.path.exists(clust_dir):
        os.makedirs(clust_dir)

    shutil.copy(file_path, clust_dir)
