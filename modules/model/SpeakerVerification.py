# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script will download pretrained models from modelscope (https://www.modelscope.cn/models)
based on the given model id, and extract embeddings from input audio. 
"""
'''
Most of the code is taken from 3D-Speaker/speakerlab/bin/infer_sv.py
My optimization:
1. Modified the way of resampling, using torchaudio.transforms.Resample, because sox_effects. apply_effects_tensor will report strange errors
2. Modified the path to download the weight output file so that it will not report an error due to insufficient permissions
3. Compared with the source code, cuda support is added, which greatly improves the inference speed
4. Added progress bar display
5. Added load.save saved embeddings to avoid double counting
'''

import os
import sys
import pathlib
import numpy as np
import torch
import torchaudio
import shutil
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

try:
    from speakerlab.process.processor import FBank
except ImportError:
    sys.path.append('%s/../..'%os.path.dirname(__file__))
    from speakerlab.process.processor import FBank

from speakerlab.utils.builder import dynamic_import

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines.util import is_official_hub_path


CAMPPLUS_VOX = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
    },
}

CAMPPLUS_COMMON = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_VOX = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2Net_huge.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_base_COMMON = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 32,
    },
}

ERes2Net_Base_3D_Speaker = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 32,
    },
}

ERes2Net_Large_3D_Speaker = {
    'obj': 'speakerlab.models.eres2net.ERes2Net.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 64,
    },
}

supports = {
    'damo/speech_campplus_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': CAMPPLUS_VOX, 
        'model_pt': 'campplus_voxceleb.bin', 
    },
    'damo/speech_campplus_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_common.bin',
    },
    'damo/speech_eres2net_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': ERes2Net_VOX,
        'model_pt': 'pretrained_eres2net.ckpt',
    },
    'damo/speech_eres2net_sv_zh-cn_16k-common': {
        'revision': 'v1.0.5', 
        'model': ERes2Net_COMMON,
        'model_pt': 'pretrained_eres2net_aug.ckpt',
    },
    'damo/speech_eres2net_base_200k_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0', 
        'model': ERes2Net_base_COMMON,
        'model_pt': 'pretrained_eres2net.pt',
    },
    'damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.1', 
        'model': ERes2Net_Base_3D_Speaker,
        'model_pt': 'eres2net_base_model.ckpt',
    },
    'damo/speech_eres2net_large_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.0', 
        'model': ERes2Net_Large_3D_Speaker,
        'model_pt': 'eres2net_large_model.ckpt',
    },
}

def move_folder_contents(source_folder, target_folder):
    os.makedirs(target_folder, exist_ok=True)
    for root, dirs, files in os.walk(source_folder):
        relative_path = os.path.relpath(root, source_folder)
        target_path = os.path.join(target_folder, relative_path)
        os.makedirs(target_path, exist_ok=True)
        for file in files:
            source_file_path = os.path.join(root, file)
            target_file_path = os.path.join(target_path, file)
            shutil.move(source_file_path, target_file_path)
    shutil.rmtree(source_folder)

def extract_embedding(wav_fpaths, Speaker_name, model_id):
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        gpu_total_memory = sum([torch.cuda.get_device_properties(i).total_memory for i in range(device_count)])
        if gpu_total_memory / (1024**3) > 3:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    local_model_dir = "pretrain"

    assert isinstance(model_id, str) and \
        is_official_hub_path(model_id), "Invalid modelscope model id."
    assert model_id in supports, "Model id not currently supported."
    save_dir = os.path.join(local_model_dir, model_id.split('/')[1])
    save_dir =  pathlib.Path(save_dir)

    conf = supports[model_id]
    # download models from modelscope according to model_id
    if not os.path.exists(os.path.join(save_dir, conf['model_pt'])):
        cache_dir = snapshot_download(
                    model_id,
                    revision=conf['revision'],
                    cache_dir='./'
                    )
        move_folder_contents(model_id.split('/')[0], local_model_dir)

    embedding_dir = pathlib.Path("output") / Speaker_name / 'embeddings'
    embedding_dir.mkdir(exist_ok=True, parents=True)

    pretrained_model = save_dir / conf['model_pt']
    pretrained_state = torch.load(pretrained_model, map_location='cpu')

    # load model
    model = conf['model']
    embedding_model = dynamic_import(model['obj'])(**model['args'])
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.to(device)  # !!!!
    embedding_model.eval()

    def load_wav(wav_file, obj_fs=16000):
        wav, fs = torchaudio.load(wav_file)
        if fs != obj_fs:
            wav = torchaudio.transforms.Resample(orig_freq=fs, new_freq=obj_fs)(wav)
        if wav.shape[0] > 1:
            wav = wav[0, :].unsqueeze(0)
        return wav

    feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)

    def compute_embedding(wav_file, save=True):

        save_path = embedding_dir / (
            '%s.npy' % (os.path.basename(wav_file).rsplit('.', 1)[0]))
        if save and save_path.exists():
            return np.load(save_path)
        
        # load wav
        wav = load_wav(wav_file)
        # compute feat
        feat = feature_extractor(wav).unsqueeze(0).to(device)
        # compute embedding
        with torch.no_grad():
            embedding = embedding_model(feat).detach().cpu().numpy()
        
        if save:
            save_path = embedding_dir / (
            '%s.npy' % (os.path.basename(wav_file).rsplit('.', 1)[0]))
            np.save(save_path, embedding)
        
        return embedding
    
    # extract embeddings
    embeds = []
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        MofNCompleteColumn(),
    )

    task_id = progress.add_task(description="Extracting embeddings...", total=len(wav_fpaths),)

    progress.start()
    for wav_file in wav_fpaths:
        embeds.append(compute_embedding(wav_file))
        progress.update(task_id, advance=1)
    progress.stop()

    return np.array(embeds).squeeze()