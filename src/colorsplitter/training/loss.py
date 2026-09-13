"""GE2E loss.

Generalized End-to-End loss for speaker verification (Wan et al., 2017), the
objective the existing checkpoints were trained with. The two learned scalars
are named ``similarity_weight`` and ``similarity_bias`` because that is exactly
how they are stored in those checkpoints — a training run resumed from one
picks them up without a translation table.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GE2ELoss"]


class GE2ELoss(nn.Module):
    """Softmax-variant GE2E loss over an ``(N, M, D)`` batch of embeddings.

    :param init_w: initial value of the learned similarity scale.
    :param init_b: initial value of the learned similarity bias.
    """

    def __init__(self, init_w: float = 10.0, init_b: float = -5.0):
        super().__init__()
        self.similarity_weight = nn.Parameter(torch.tensor([init_w], dtype=torch.float32))
        self.similarity_bias = nn.Parameter(torch.tensor([init_b], dtype=torch.float32))

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """``embeds``: ``(N, M, D)`` utterance embeddings, normalised here.

        The reference implementations normalise inside the loss rather than
        trusting the caller, and this one does too: a caller that forgets would
        otherwise get silently wrong centroids, which is exactly the kind of
        mistake that is invisible in the loss curve.
        """
        if embeds.dim() != 3:
            raise ValueError(f"expected (N, M, D) embeddings, got {tuple(embeds.shape)}")
        n_speakers, n_utterances, dim = embeds.shape
        if n_speakers < 2:
            raise ValueError("GE2E needs at least 2 classes per batch")
        if n_utterances < 2:
            raise ValueError("GE2E needs at least 2 utterances per class")

        device = embeds.device
        embeds = F.normalize(embeds, dim=2)
        n_all = n_speakers * n_utterances
        flat = embeds.reshape(n_all, dim)
        index = torch.arange(n_speakers, device=device)

        # Inclusive centroid: mean of every utterance of the class. One matmul
        # gives every utterance's similarity to every centroid.
        centroids_incl = F.normalize(embeds.mean(dim=1), dim=1)  # (N, D)
        sim = self.similarity_weight * (flat @ centroids_incl.T) + self.similarity_bias
        sim = sim.reshape(n_speakers, n_utterances, n_speakers)

        # Exclusive centroid: leave-one-out mean, used for the true class so the
        # model cannot satisfy the objective by copying its own input.
        total = embeds.sum(dim=1, keepdim=True)  # (N, 1, D)
        centroids_excl = F.normalize((total - embeds) / (n_utterances - 1), dim=2)
        centroids_excl = centroids_excl.reshape(n_all, dim)
        sim_excl_self = self.similarity_weight * (flat * centroids_excl).sum(dim=1)
        sim_excl_self = sim_excl_self + self.similarity_bias  # (N * M,)

        sim[index, :, index] = sim_excl_self.reshape(n_speakers, n_utterances)

        log_prob = sim - torch.logsumexp(sim, dim=2, keepdim=True)
        return -log_prob[index, :, index].mean()
