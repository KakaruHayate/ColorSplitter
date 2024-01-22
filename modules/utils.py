from resemblyzer import preprocess_wav
from modules.model.voice_encoder import VoiceEncoder
from tqdm import tqdm
import numpy as np
import pickle
import os
import importlib

class GetEmbeds:
    """ Used to obtain embedding vectors for audio. Directly input wav.
    """

    def __init__(self, encoder_type, Speaker_name):
        self.encoder_type = encoder_type
        self.Speaker_name = Speaker_name
        if self.encoder_type == 'timbre':
            self.encoder = VoiceEncoder(weights_fpath="pretrain/encoder_1570000.bak")
        elif self.encoder_type == 'emotion':
            self.emotion_module = importlib.import_module('modules.model.emotion_encoder')
        elif self.encoder_type == 'mix':
            self.encoder = VoiceEncoder(weights_fpath="pretrain/encoder_1570000.bak")
            self.emotion_module = importlib.import_module('modules.model.emotion_encoder')
        else:
            raise ValueError(
                '%s is not currently supported.' % self.encoder_type
            )

    def __call__(self, wav_fpaths):
        if self.encoder_type == 'timbre':
            embeds = self.timbre_encoder(wav_fpaths)
        if self.encoder_type == 'emotion':
            embeds = self.emotion_encoder(wav_fpaths)
        if self.encoder_type == 'mix':
            embeds = self.mix_encoder(wav_fpaths)
        
        return embeds
    
    def timbre_encoder(self, wav_fpaths):
        features_path = os.path.join("input", self.Speaker_name, "features(timbre).pkl")
        # Check if features already exist
        if os.path.exists(features_path):
            with open(features_path, 'rb') as f:
                embeds = pickle.load(f)
        else:
            wavs = [preprocess_wav(wav_fpath) for wav_fpath in \
                    tqdm(wav_fpaths, f"Preprocessing wavs ({len(wav_fpaths)} utterances)")] 
            embeds = np.array(list(map(self.encoder.embed_utterance, wavs)))
            with open(features_path, 'wb') as f:
                pickle.dump(embeds, f)
        
        return embeds
    
    def emotion_encoder(self, wav_fpaths):
        features_path = os.path.join("input", self.Speaker_name, "features(emotion).pkl")
        # Check if features already exist
        if os.path.exists(features_path):
            with open(features_path, 'rb') as f:
                embeds = pickle.load(f)
        else:
            embeds = [self.emotion_module.extract_wav(wav_fpath) for wav_fpath in \
                    tqdm(wav_fpaths, f"Preprocessing wavs ({len(wav_fpaths)} utterances)")] 
            embeds = np.concatenate(embeds,axis=0)
            with open(features_path, 'wb') as f:
                pickle.dump(embeds, f)
        
        return embeds
    
    def mix_encoder(self, wav_fpaths):
        features_path = os.path.join("input", self.Speaker_name, "features(mix).pkl")
        # Check if features already exist
        if os.path.exists(features_path):
            with open(features_path, 'rb') as f:
                embeds = pickle.load(f)
        else:
            timber_embeds = self.timbre_encoder(wav_fpaths)
            emotion_embeds = self.emotion_encoder(wav_fpaths)
            embeds = np.concatenate((timber_embeds, emotion_embeds), axis=1)
            with open(features_path, 'wb') as f:
                pickle.dump(embeds, f)
        
        return embeds
        