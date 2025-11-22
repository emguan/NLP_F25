#!/usr/bin/env python3

# CS465 at Johns Hopkins University.

# Subclass ConditionalRandomFieldBackprop to get a biRNN-CRF model.

from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
from math import inf, log, exp
from pathlib import Path
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import Tensor, cuda
from jaxtyping import Float

from corpus import IntegerizedSentence, Sentence, Tag, TaggedCorpus, Word
from integerize import Integerizer
from crf_backprop import ConditionalRandomFieldBackprop, TorchScalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

class ConditionalRandomFieldNeural(ConditionalRandomFieldBackprop):
    """A CRF that uses a biRNN to compute non-stationary potential
    matrices.  The feature functions used to compute the potentials
    are now non-stationary, non-linear functions of the biRNN
    parameters."""

    neural = True    # class attribute that indicates that constructor needs extra args
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 rnn_dim: int,
                 unigram: bool = False):
        # [doctring inherited from parent method]

        if unigram:
            raise NotImplementedError("Not required for this homework")

        self.rnn_dim = rnn_dim
        self.e = lexicon.size(1) # dimensionality of word's embeddings
        self.E = lexicon

        nn.Module.__init__(self)  
        super().__init__(tagset, vocab, unigram)


    @override
    def init_params(self) -> None:

        """
            Initialize all the parameters you will need to support a bi-RNN CRF
            This will require you to create parameters for M, M', U_a, U_b, theta_a
            and theta_b. Use xavier uniform initialization for the matrices and 
            normal initialization for the vectors. 
        """

        # See the "Parameterization" section of the reading handout to determine
        # what dimensions all your parameters will need.

        k = self.k
        e = self.e
        r = self.rnn_dim

        self.M = nn.Parameter(torch.empty(r, e))
        self.M_prime = nn.Parameter(torch.empty(r, r))

        trans_hidden_dim = r 
        trans_feat_dim = 2 * r + 2 * k

        self.U_a = nn.Parameter(torch.empty(trans_hidden_dim, trans_feat_dim))
        self.theta_a = nn.Parameter(torch.empty(trans_hidden_dim))
        
        emit_hidden_dim = r
        emit_feat_dim = 2 * r + k + e

        self.U_b = nn.Parameter(torch.empty(emit_hidden_dim, emit_feat_dim))
        self.theta_b = nn.Parameter(torch.empty(emit_hidden_dim))

        for W in (self.M, self.M_prime, self.U_a, self.U_b):
            nn.init.xavier_uniform_(W)

        nn.init.normal_(self.theta_a, mean=0.0, std=0.1)
        nn.init.normal_(self.theta_b, mean=0.0, std=0.1)

        self._h_fwd = None
        self._h_bwd = None
        self._h_ctx = None

    @override
    def init_optimizer(self, lr: float, weight_decay: float) -> None:
        # [docstring will be inherited from parent]
    
        # Use AdamW optimizer for better training stability
        self.optimizer = torch.optim.AdamW( 
            params=self.parameters(),       
            lr=lr, weight_decay=weight_decay
        )                                   
        self.scheduler = None            
       
    @override
    def updateAB(self) -> None:
        # Nothing to do - self.A and self.B are not used in non-stationary CRFs
        pass

    @override
    def setup_sentence(self, isent: IntegerizedSentence) -> None:
        """Pre-compute the biRNN prefix and suffix contextual features (h and h'
        vectors) at all positions, as defined in the "Parameterization" section
        of the reading handout.  They can then be accessed by A_at() and B_at().
        
        Make sure to call this method from the forward_pass, backward_pass, and
        Viterbi_tagging methods of HiddenMarkovMOdel, so that A_at() and B_at()
        will have correct precomputed values to look at!"""

        device = next(self.parameters()).device
        r = self.rnn_dim
        n = len(isent)

        word_ids = torch.tensor([w for (w, _) in isent], device=device, dtype=torch.long)
        X = self.E[word_ids].to(device)

        h_fwd = torch.zeros(n, r, device=device)
        for j in range(1, n):
            h_fwd[j] = torch.tanh(
                X[j] @ self.M.T + h_fwd[j - 1] @ self.M_prime.T
            )

        h_bwd = torch.zeros(n, r, device=device)
        for j in range(n - 2, -1, -1):
            h_bwd[j] = torch.tanh(
                X[j] @ self.M.T + h_bwd[j + 1] @ self.M_prime.T
            )

        h_ctx = torch.cat([h_fwd, h_bwd], dim=1)

        self._h_fwd = h_fwd
        self._h_bwd = h_bwd
        self._h_ctx = h_ctx

    @override
    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        isent = self._integerize_sentence(sentence, corpus)
        super().accumulate_logprob_gradient(sentence, corpus)

    @override
    @typechecked
    def A_at(self, position, sentence) -> Tensor:
        
        """Computes non-stationary k x k transition potential matrix using biRNN 
        contextual features and tag embeddings (one-hot encodings). Output should 
        be ϕA from the "Parameterization" section in the reading handout."""

        device = next(self.parameters()).device
        k = self.k

        h_ctx_j = self._h_ctx[position].to(device)

        tag_eye = self.eye.to(device)

        h_rep = h_ctx_j.view(1, 1, -1).expand(k, k, -1)

        e_s = tag_eye.view(k, 1, k).expand(k, k, k)

        e_t = tag_eye.view(1, k, k).expand(k, k, k)

        feats = torch.cat([h_rep, e_s, e_t], dim=2)
        feats_flat = feats.view(k * k, -1)

        hidden = torch.tanh(feats_flat @ self.U_a.T)

        scores = hidden @ self.theta_a
        scores = scores.view(k, k)    

        A = torch.exp(scores)

        mask = torch.ones_like(A)
        mask[:, self.bos_t] = 0.0
        mask[self.eos_t, :] = 0.0

        return A * mask   

    @override
    @typechecked
    def B_at(self, position, sentence) -> Tensor:
        """Computes non-stationary k x V emission potential matrix using biRNN 
        contextual features, tag embeddings (one-hot encodings), and word embeddings. 
        Output should be ϕB from the "Parameterization" section in the reading handout."""

        device = next(self.parameters()).device
        k = self.k

        h_ctx_j = self._h_ctx[position].to(device)

        tag_eye = self.eye.to(device)

        h_rep = h_ctx_j.view(1, 1, -1).expand(k, k, -1)

        e_s = tag_eye.view(k, 1, k).expand(k, k, k)

        e_t = tag_eye.view(1, k, k).expand(k, k, k)

        feats = torch.cat([h_rep, e_s, e_t], dim=2)
        feats_flat = feats.view(k * k, -1)

        hidden = torch.tanh(feats_flat @ self.U_a.T)

        scores = hidden @ self.theta_a
        scores = scores.view(k, k)    

        A = torch.exp(scores)

        mask = torch.ones_like(A)
        mask[:, self.bos_t] = 0.0
        mask[self.eos_t, :] = 0.0

        return A * mask  