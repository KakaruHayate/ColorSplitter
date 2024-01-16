import torch
import numpy as np
from modules.emotion_encoder.emotion_extract import *

class EmotionEncoder:
    def __init__(self, model_name='audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'):
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = EmotionModel.from_pretrained(model_name).to(self.device)

    def encode(self, wav, sr, embeddings=False):
        y = self.processor(wav, sampling_rate=sr)
        y = y['input_values'][0]
        y = torch.from_numpy(y).to(self.device)

        with torch.no_grad():
            y = self.model(y)[0 if embeddings else 1]

        y = y.detach().cpu().numpy()
        return y