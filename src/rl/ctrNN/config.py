"""Shared training/model configuration for the ctrNN package.

Single source of truth for constants that must stay in sync across
train.py, jdh.py, and decoder_model.py. Import these instead of
redefining them locally.
"""

# --- data / training ---
BLOCK_SIZE = 512        # context chunk length
TARGET_SIZE = 8         # target chunk length (must match Predictor.t_tgt)
BATCH_SIZE = 32
MAX_ITERS = 400
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 100
EMA_DECAY = 0.99        # momentum of the target encoder EMA update
L_RELEVANCE = [1,1,0.05]  # [iLoss, vLoss, cLoss] relevance weights for the loss function