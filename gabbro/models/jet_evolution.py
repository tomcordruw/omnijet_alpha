import logging
import time
from typing import Any, Dict, Tuple

import awkward as ak
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vector
from tqdm import tqdm
from collections import Counter

from gabbro.metrics.utils import calc_accuracy
from gabbro.models.gpt_model import BackboneModel

vector.register_awkward()

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# ------------ BACKBONE + Jet Evolution (token-pair prediction) head ---------
# ----------------------------------------------------------------------------

class LeftHead(nn.Module):
    """Unchanged original head"""
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        return self.fc(x)
        

class RightHead(nn.Module):
    """Right head with cross attention"""
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, memory):
        if memory is not None:
            attention_output, _ = self.cross_attn(query=x, key=memory, value=memory)
            x = self.norm(x + attention_output)
        else:
            x = self.norm(x)
        return self.fc(x)
    

class BackboneTokenPairPredictionLightning(L.LightningModule):
    """PyTorch Lightning module for training the backbone model."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        model_kwargs={},
        token_dir=None,
        verbose=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        # initialize the backbone
        self.module = BackboneModel(**model_kwargs)

        # initialize the left model head
        self.left_head = LeftHead(
            embedding_dim=model_kwargs["embedding_dim"],
            vocab_size=model_kwargs["vocab_size"],
        )

        # initialize the right model head
        self.right_head = RightHead(
            embedding_dim=model_kwargs["embedding_dim"],
            vocab_size=model_kwargs["vocab_size"],
        )

        # Optional: load token weights to account for imbalanced token distributions
        self.token_weights_path = model_kwargs.get("token_weights_path", "None")
        print(f"Token frequency weights path: {self.token_weights_path}")
        
        self.token_weights = None
        if self.token_weights_path is not None:
            if self.token_weights_path != "None":
                self.load_token_weights(self.token_weights_path)

        # Create the loss function (can use the loaded token weights)
        self.criterion = nn.CrossEntropyLoss(
            weight=self.token_weights,
            ignore_index=0,
            reduction='none'
        )

        self.token_dir = token_dir
        self.verbose = verbose

        self.train_loss_history = []
        self.val_loss_list = []

        self.validation_cnt = 0
        self.validation_output = {}

        self.backbone_weights_path = model_kwargs.get("backbone_weights_path", "None")

        print(f"Backbone weights path: {self.backbone_weights_path}")

        if self.backbone_weights_path is not None:
            if self.backbone_weights_path != "None":
                self.load_backbone_weights(self.backbone_weights_path)

    def load_backbone_weights(self, ckpt_path):
        print(f"Loading backbone weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        self.load_state_dict(state_dict, strict=False)

    def load_token_weights(self, weights_path):
        print(f"Loading token frequency weights from {weights_path}")
        weights = torch.load(weights_path)
        self.token_weights = weights.to(self.device)

    def forward(self, x, mask=None, mode="both"):
        if self.module.return_embeddings:
            if mode == "both":
                # Use the gpt-backbone model
                backbone_out = self.module(x, mask)

                # Create indices corresponding to left children, 1, 4, 7,...
                # and right children, 2, 5, 8,...
                left_idxs = torch.arange(1, x.shape[1], 3, device=x.device)
                right_idxs = torch.arange(2, x.shape[1], 3, device=x.device)

                # Ensure that the number of indices match (truncate to max_len)
                max_len = x.shape[1] // 3

                # Get the backbone output embeddings for left/right children
                left_input = backbone_out[:, left_idxs[:max_len], :]
                right_input = backbone_out[:, right_idxs[:max_len], :]

                # Use the left head with the left children's embeddings
                left_logits = self.left_head(left_input)

                # Use the right head with the right children's embeddings and
                # left children embeddings as memory for cross attention
                right_logits = self.right_head(right_input, left_input.detach())

                return torch.stack([left_logits, right_logits], dim=2)
            
            elif mode == "left":
                backbone_out = self.module(x, mask)
                return self.left_head(backbone_out)
            
            elif mode == "right":
                backbone_out = self.module(x, mask)
                return self.right_head(backbone_out, memory=None)

        else:
            logits = self.module(x, mask)
        if self.verbose:
            print("Logits shape: ", logits.shape)
        return logits

    def model_step(self, batch, return_logits=False):
        """Perform a single model step on a batch of data.

        Parameters
        ----------
        batch : dict
            A batch of data as a dictionary containing the input and target tensors,
            as well as the mask.
        return_logits : bool, optional
            Whether to return the logits or not. (default is False)
        """

        # Input is the sequence of tokens splitting, after a jet is clustered via fastjet
        # i.e. the unique_history_order but in reversed
        # targets_left are the first product of the split
        # targets_right are the second product of the split
        # i.e. it splits like a binary tree at every point
        X = batch["part_features"]
        mask = batch["part_mask"]
        X = X.squeeze().long()
        input = X[:, :, 1]
        
        # Divide by three
        max_length = (input.shape[1] // 3)
        
        # Create and stack the left and right targets
        targets_left = X[:, 1::3, 1][:, :max_length]
        targets_right = X[:, 2::3, 1][:, :max_length]
        targets = torch.stack([targets_left, targets_right], dim=2)

        # compute the logits (i.e. the predictions for the next token)
        # logits in the shape [batch_size * jet sequence length * 2 (left and right) * codebook size]
        logits = self.forward(input, mask)

        # Mask zero-padded values
        pair_mask = (targets != 0).all(dim=2)
        
        # Apply mask to logits and targets
        # B=Batches, T=Tokens , P=Positions (left/right), C=Classes
        logits = logits[pair_mask] # B, T, P, C => B * T, P, C
        targets = targets[pair_mask] # B, T, P => B * T, P

        # Reshape the logits and targets to work with the loss function
        N, P, C = logits.shape
        logits = logits.view(N * P, C)
        targets = targets.contiguous().view(N * P)

        # Get the logits and targets for each respective head
        logits_left = logits[::2]
        logits_right = logits[1::2]
        targets_left = targets[::2]
        targets_right = targets[1::2]

        # Calculate losses for both permutations of predictions
        # Predictions [left, right] -> Targets [left, right]
        # Predictions [left, right] -> Targets [right, left]
        loss_lr = self.criterion(logits_left, targets_left) + self.criterion(logits_right, targets_right)
        loss_rl = self.criterion(logits_left, targets_right) + self.criterion(logits_right, targets_left)

        # Minimum loss from both permutations
        loss = torch.minimum(loss_lr, loss_rl)
        loss = loss.sum() / N
        
        # Penalty for predicting the same token twice
        # Mean value for all predictions, if True -> 1.0, false -> 0.0
        preds_left = logits_left.argmax(dim=1)
        preds_right = logits_right.argmax(dim=1)
        penalty = (preds_left == preds_right).float().mean()
        loss = loss + 0.1 * penalty

        if return_logits:
            return loss, X, logits, mask, targets

        return loss
        
    @torch.no_grad()
    def predict(self, tokens, mode="both", device=None):
        """Perform predictions on input tokens.

        Parameters
        ----------
        tokens : ak.Array, torch.Tensor, int
            A sequence of integer tokens for prediction 
        mode : str, optional
            Whether to use both heads for the prediction. (default is both)
        """

        if device is None:
            device = next(self.module.parameters()).device
        # Convert input to list
        if isinstance(tokens, ak.Array):
            tokens = tokens.tolist()
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        elif isinstance(tokens, int):
            tokens = [tokens]

        # Use a single head to predict 
        def predict_token(input_seq, head, true_token=None):
            input_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)
            logits = self.forward(input_tensor, mode=head)
            # Only look at the final predicted token
            probs = F.softmax(logits[:, -1, :], dim=-1)
            # Exclude start token from predicted classes
            token = torch.multinomial(probs[:, 1:], num_samples=1) + 1
            # Get the top-k tokens (was used for testing)
            topk = torch.topk(probs[:, 1:], 5)
            print(f"\n{head} head prediction: {token.item()}", end="")
            return token.item()

        # Predict the last two tokens of a sequence based on the parent
        # Requires sequences with at least 3 tokens ([..., parent, left, right])
        if mode == "both" and len(tokens) >= 3:
            print(f"\n--- Predicting both children for the following sequence: {tokens[:-2]}")
            # Make left prediction based on sequence up to the 3rd last (which should have the parent)
            left_pred = predict_token(tokens[:-2], "left", tokens[-2])
            # Make right prediction based on the same sequence + the predicted left child
            right_pred = predict_token(tokens[:-2] + [left_pred], "right", tokens[-1])

            # Compare the actual sequence and predicted sequence
            print(f"\n\nGround truth:\nParent: {tokens[-3]} => left: {tokens[-2]}, right: {tokens[-1]}")
            predicted_sequence = ak.Array(tokens[:-2] + [left_pred, right_pred])
            print(f"\nPredicted sequence: {predicted_sequence}")
            print(f"\nActual sequence: {tokens}")
            return left_pred, right_pred

        else:
            # Predict both children from a single token
            print(f"\n--- Predicting both children of a single token: {tokens} ---")
            left_pred = predict_token(tokens, "left")
            right_pred = predict_token(tokens + [left_pred], "right")
            print(f"\nParent: {tokens[-1]} => left: {left_pred}, right: {right_pred}")
            return left_pred, right_pred

    @torch.no_grad()
    def generate_batch(self, batch_size):
        codebook_size = self.module.vocab_size - 1
        device = next(self.module.parameters()).device  # get the device of the model

        # TODO: Currently starts sequences with 0, could sample from a distribution of tokens instead
        idx = torch.zeros(batch_size, 1).long().to(device) # B * T
        seq_length = self.module.max_sequence_len
        # Offset for appending child tokens as parents
        # TODO: This results in trees with breadth-first predictions
        # but training data seems to follow depth-first => should be compatible with multiple strategies
        offset = 0

        # Generate a sequence of tokens using both heads
        for i in range(0, seq_length//3):
            # Feed current sequence into left head to get prediction for left child
            logits_left = self(idx, mode="left")[:, -1, :]
            # Apply softmax and sample token to append (skip index 0 if it's reserved for start token)
            probs_left = F.softmax(logits_left[:, 1:], dim=-1) # B * P * C
            tokens_left = torch.multinomial(probs_left, num_samples=1) + 1
            # Append token to sequence
            idx = torch.cat((idx,tokens_left), dim=1)

            # Repeat the same for the right head
            logits_right = self(idx, mode="right")[:, -1, :]
            probs_right = F.softmax(logits_right[:, 1:], dim=-1) # B * P * C
            tokens_right = torch.multinomial(probs_right, num_samples=1) + 1
            idx = torch.cat((idx,tokens_right), dim=1)

            # Every two children, skip the parent
            if i % 2 == 0:
                offset += 1
            # Append the token that will be the next parent in the sequence to predict from
            idx_parent_nxt = idx[:, i+offset].view(-1, 1)
            idx = torch.cat((idx, idx_parent_nxt), dim=1)

        # Pad with stop tokens at the end
        last_index = (idx.size(1) // 3) * 3
        jets_trimmed = idx[:, :last_index]
        padding_amount = seq_length - jets_trimmed.size(1)

        gen_batch_np = idx.detach().cpu().numpy()
        gen_batch_ak = ak.from_numpy(gen_batch_np)
        gen_batch_until_stop = []

        # Loop over the jets in the batch, and only keep the tokens until the (first) stop token
        # Should be reworked for generating trees with more depth
        for jet in gen_batch_ak:
            stop_token_position = np.where(jet == self.module.vocab_size - 1)
            if len(stop_token_position[0]) > 0:
                stop_token_position = stop_token_position[0][0]
            else:
                stop_token_position = jet.shape[0]
            gen_batch_until_stop.append(jet[:stop_token_position])

        return ak.Array(gen_batch_until_stop)

    def generate_n_jets_batched(self, n_jets, batch_size, saveas=None):
        """Generate jets in batches.

        Parameters
        ----------
        n_jets : int
            Number of jets to generate.
        batch_size : int
            Batch size to use during generation (use as large as possible with memory.)
        saveas : str, optional
            Path to save the generated jets to (in parquet format). (default is None)

        Returns
        -------
        ak.Array
            The generated jets (i.e. their token ids, in the shape (n_jets, <var>).
        """
        n_batches = n_jets // batch_size + 1
        generated_jets = []

        print(f"Generating {n_jets} jets in {n_batches} batches of size {batch_size}")

        for i in tqdm(range(n_batches)):
            gen_batch_ak = self.generate_batch(batch_size)
            generated_jets.append(gen_batch_ak)

        # concatenate the generated batches
        generated_jets = ak.concatenate(generated_jets)[:n_jets]

        if saveas is not None:
            print(f"Saving generated jets to {saveas}")
            ak.to_parquet(generated_jets, saveas)

        return generated_jets

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set."""
        loss = self.model_step(batch)

        self.train_loss_history.append(float(loss))
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_start(self) -> None:
        self.preprocessing_dict = (
            self.trainer.datamodule.hparams.dataset_kwargs_common.feature_dict
        )

    def on_train_epoch_start(self):
        logger.info(f"Epoch {self.trainer.current_epoch} starting.")
        self.epoch_train_start_time = time.time()  # start timing the epoch

    def on_train_epoch_end(self):
        self.epoch_train_end_time = time.time()
        self.epoch_train_duration_minutes = (
            self.epoch_train_end_time - self.epoch_train_start_time
        ) / 60
        self.log(
            "epoch_train_duration_minutes",
            self.epoch_train_duration_minutes,
            on_epoch=True,
            prog_bar=False,
        )
        logger.info(
            f"Epoch {self.trainer.current_epoch} finished in"
            f" {self.epoch_train_duration_minutes:.1f} minutes."
        )

    def on_train_end(self):
        pass

    def on_validation_epoch_start(self) -> None:
        self.val_token_ids_list = []
        self.val_token_masks_list = []

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, X, logits, mask, targets = self.model_step(batch, return_logits=True)

        self.val_token_ids_list.append(batch["part_features"].float().detach().cpu().numpy())
        self.val_token_masks_list.append(batch["part_mask"].float().detach().cpu().numpy())
        self.log("val_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_test_epoch_start(self) -> None:
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, X, logits, mask, targets = self.model_step(batch, return_logits=True)
        self.log("test_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        pass

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training."""
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}