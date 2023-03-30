from typing import Iterable, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from omegaconf import OmegaConf

import math
from functools import partial
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


from .layers import ViTEncoder, ViTDecoder


class BaseQuantizer(nn.Module):
    def __init__(self, embed_dim: int, n_embed: int, straight_through: bool = True, use_norm: bool = True,
                 use_residual: bool = False, num_quantizers: Optional[int] = None) -> None:
        super().__init__()
        self.straight_through = straight_through
        self.norm = lambda x: F.normalize(x, dim=-1) if use_norm else x

        self.use_residual = use_residual
        self.num_quantizers = num_quantizers

        self.embed_dim = embed_dim
        self.n_embed = n_embed

        self.embedding = nn.Embedding(self.n_embed, self.embed_dim)
        self.embedding.weight.data.normal_()
        
    def quantize(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        pass
    
    def forward(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        if not self.use_residual:
            z_q, loss, encoding_indices = self.quantize(z)
        else:
            z_q = torch.zeros_like(z)
            residual = z.detach().clone()

            losses = []
            encoding_indices = []

            for _ in range(self.num_quantizers):
                z_qi, loss, indices = self.quantize(residual.clone())
                residual.sub_(z_qi)
                z_q.add_(z_qi)

                encoding_indices.append(indices)
                losses.append(loss)

            losses, encoding_indices = map(partial(torch.stack, dim = -1), (losses, encoding_indices))
            loss = losses.mean()

        # preserve gradients with straight-through estimator
        if self.straight_through:
            z_q = z + (z_q - z).detach()

        return z_q, loss, encoding_indices


class VectorQuantizer(BaseQuantizer):
    def __init__(self, embed_dim: int, n_embed: int, beta: float = 0.25, use_norm: bool = True,
                 use_residual: bool = False, num_quantizers: Optional[int] = None, **kwargs) -> None:
        super().__init__(embed_dim, n_embed, True,
                         use_norm, use_residual, num_quantizers)
        
        self.beta = beta

    def quantize(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        z_reshaped_norm = self.norm(z.view(-1, self.embed_dim))
        embedding_norm = self.norm(self.embedding.weight)
        
        d = torch.sum(z_reshaped_norm ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_norm ** 2, dim=1) - 2 * \
            torch.einsum('b d, n d -> b n', z_reshaped_norm, embedding_norm)

        encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        encoding_indices = encoding_indices.view(*z.shape[:-1])
        
        z_q = self.embedding(encoding_indices).view(z.shape)
        z_qnorm, z_norm = self.norm(z_q), self.norm(z)
        
        # compute loss for embedding
        loss = self.beta * torch.mean((z_qnorm.detach() - z_norm)**2) +  \
               torch.mean((z_qnorm - z_norm.detach())**2)

        return z_qnorm, loss, encoding_indices


class ViTVQ(pl.LightningModule):
    def __init__(self, image_key: str, image_size: int, patch_size: int, encoder: OmegaConf, decoder: OmegaConf, quantizer: OmegaConf,
                 loss: OmegaConf, path: Optional[str] = None, ignore_keys: List[str] = list(), scheduler: Optional[OmegaConf] = None) -> None:
        super().__init__()
        self.path = path
        self.ignore_keys = ignore_keys 
        self.image_key = image_key
        self.scheduler = scheduler 
        
        self.loss = initialize_from_config(loss)
        self.encoder = ViTEncoder(image_size=image_size, patch_size=patch_size, **encoder)
        self.decoder = ViTDecoder(image_size=image_size, patch_size=patch_size, **decoder)
        self.quantizer = VectorQuantizer(**quantizer)
        self.pre_quant = nn.Linear(encoder.dim, quantizer.embed_dim)
        self.post_quant = nn.Linear(quantizer.embed_dim, decoder.dim)

        if path is not None:
            self.init_from_ckpt(path, ignore_keys)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:    
        quant, diff = self.encode(x)
        dec = self.decode(quant)
        
        return dec, diff

    def init_from_ckpt(self, path: str, ignore_keys: List[str] = list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        
    def encode(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        h = self.encoder(x)
        h = self.pre_quant(h)
        quant, emb_loss, _ = self.quantizer(h)
        
        return quant, emb_loss

    def decode(self, quant: torch.FloatTensor) -> torch.FloatTensor:
        quant = self.post_quant(quant)
        dec = self.decoder(quant)
        
        return dec

    def encode_codes(self, x: torch.FloatTensor) -> torch.LongTensor:
        h = self.encoder(x)
        h = self.pre_quant(h)
        _, _, codes = self.quantizer(h)
        
        return codes

    def decode_codes(self, code: torch.LongTensor) -> torch.FloatTensor:
        quant = self.quantizer.embedding(code)
        quant = self.quantizer.norm(quant)
        
        if self.quantizer.use_residual:
            quant = quant.sum(-2)  
            
        dec = self.decode(quant)
        
        return dec

    def get_input(self, batch: Tuple[Any, Any], key: str = 'image') -> Any:
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.dtype == torch.double:
            x = x.float()

        return x.contiguous()

    def training_step(self, batch: Tuple[Any, Any], batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencoder
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step, batch_idx,
                                            last_layer=self.decoder.get_last_layer(), split="train")

            self.log("train/total_loss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            del log_dict_ae["train/total_loss"]
            
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step, batch_idx,
                                                last_layer=self.decoder.get_last_layer(), split="train")
            
            self.log("train/disc_loss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            del log_dict_disc["train/disc_loss"]
            
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            
            return discloss

    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> Dict:
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, batch_idx,
                                        last_layer=self.decoder.get_last_layer(), split="val")

        rec_loss = log_dict_ae["val/rec_loss"]

        self.log("val/rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/total_loss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        del log_dict_ae["val/rec_loss"]
        del log_dict_ae["val/total_loss"]

        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        if hasattr(self.loss, 'discriminator'):
            discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step, batch_idx,
                                                last_layer=self.decoder.get_last_layer(), split="val")
            
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        
        return self.log_dict

    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate
        optim_groups = list(self.encoder.parameters()) + \
                       list(self.decoder.parameters()) + \
                       list(self.pre_quant.parameters()) + \
                       list(self.post_quant.parameters()) + \
                       list(self.quantizer.parameters())
        
        optimizers = [torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)]
        schedulers = []
        
        if hasattr(self.loss, 'discriminator'):
            optimizers.append(torch.optim.AdamW(self.loss.discriminator.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4))

        if self.scheduler is not None:
            self.scheduler.params.start = lr
            scheduler = initialize_from_config(self.scheduler)
            
            schedulers = [
                {
                    'scheduler': lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                } for optimizer in optimizers
            ]
   
        return optimizers, schedulers
        
    def log_images(self, batch: Tuple[Any, Any], *args, **kwargs) -> Dict:
        log = dict()
        x = self.get_input(batch, self.image_key).to(self.device)
        quant, _ = self.encode(x)
        
        log["originals"] = x
        log["reconstructions"] = self.decode(quant)
        
        return 