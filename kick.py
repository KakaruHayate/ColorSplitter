import os
import shutil
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--spk', type=str, help='Speaker name')
parser.add_argument('--clust', type=int, help='Cluster value')

args = parser.parse_args()

Speaker_name = args.spk #Speaker name
clust_value = args.clust # Cluster value

data = pd.read_csv(os.path.join('output', Speaker_name, f'clustered_files(timbre).csv'))

for index, row in data.iterrows():
    file_path = row['filename']
    clust = row['clust']

    if clust == clust_value:
        clust_dir = os.path.join('input', f'{Speaker_name}_{clust_value}')
        if not os.path.exists(clust_dir):
            os.makedirs(clust_dir)

        shutil.move(file_path, clust_dir)
