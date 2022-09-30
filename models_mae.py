# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from resnet import resnet101,resnet50
import torch.nn.functional as F


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.middle_norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed_last = nn.Linear(embed_dim, 512, bias=True)
        self.decoder_embed_mid = nn.Linear(embed_dim, 512, bias=True)

        self.mask_token_last = nn.Parameter(torch.zeros(1, 1, 512))
        self.mask_token_mid = nn.Parameter(torch.zeros(1, 1, 512))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, 512), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks_last = nn.ModuleList([
            Block(512, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(2)])
        self.decoder_blocks_mid = nn.ModuleList([
            Block(512, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(2)])

        self.decoder_norm_last = norm_layer(512)
        self.decoder_norm_mid = norm_layer(512)
        self.decoder_pred_last = nn.Linear(512, 2048, bias=True) # decoder to patch
        self.decoder_pred_mid = nn.Linear(512, 4096, bias=True)
        self.last_to_mid = nn.Linear(768, 768,bias=True)
        self.mid_to_last = nn.Linear(768, 768,bias=True)
        # --------------------------------------------------------------------------
    
        self.norm_pix_loss = norm_pix_loss
        # self.resnet50=resnet50(pretrained=True)
        self.resnet50=resnet50(pretrained=True)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token_last, std=.02)
        torch.nn.init.normal_(self.mask_token_mid, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk_i,blk in enumerate(self.blocks):
            x = blk(x)
            if blk_i==5:
                middle_x = x.clone()
        x = self.norm(x)
        middle_x = self.norm(middle_x)
        return x, middle_x, mask, ids_restore


    

    def forward_decoder_last(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed_last(x)
        mask_tokens = self.mask_token_last.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # add pos embed
        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks_last:
            x = blk(x)
        x = self.decoder_norm_last(x)
        # predictor projection
        x = self.decoder_pred_last(x)
        # remove cls token
        x = x[:, 1:, :]
        return x



    def forward_decoder_mid(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed_mid(x)
        mask_tokens = self.mask_token_mid.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # add pos embed
        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks_mid:
            x = blk(x)
        x = self.decoder_norm_mid(x)
        # predictor projection
        x = self.decoder_pred_mid(x)
        # remove cls token
        x = x[:, 1:, :]
        return x

    def forward_loss_r(self, target, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        n,l,d=target.size()
        mean=target.mean(dim=[0,1],keepdim=True)
        var = target.var(dim=[0,1], keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5
        target=target.reshape(n,l,d)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75 , epoch=0):
        n,_,_,_=imgs.shape
        with torch.no_grad():
            self.resnet50.eval()
            resnet_latent_mid,resnet_latent_last=self.resnet50(imgs)
            resnet_latent_mid,resnet_latent_last=resnet_latent_mid.detach(),resnet_latent_last.detach()
            resnet_latent_last = F.interpolate(resnet_latent_last, (14,14), mode="nearest")

            Y = torch.zeros((n,4096,14,14)).to(resnet_latent_mid.device)
            for i in range(Y.shape[2]):
                for j in range(Y.shape[3]):
                    k=resnet_latent_mid[:,:,4*i:(4*i+4),4*j:(4*j+4)]
                    k=k.reshape(n,-1)
                    Y[:, :, i, j] = k
            resnet_latent_mid = Y

            resnet_latent_mid,resnet_latent_last=resnet_latent_mid.permute(0,2,3,1),resnet_latent_last.permute(0,2,3,1)
            n,w,h,d=resnet_latent_mid.size()
            resnet_latent_mid=resnet_latent_mid.reshape(shape=(-1,w*h,4096))
            resnet_latent_last=resnet_latent_last.reshape(shape=(-1,w*h,2048))

        latent,mid_latent,mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred_last= self.forward_decoder_last(latent,ids_restore)  # [N, L, p*p*3]
        mid_latent_sigmoid = F.sigmoid(self.last_to_mid(mid_latent))
        latent_sigmoid = F.sigmoid(self.mid_to_last(latent)) 
        mid_latent = mid_latent * mid_latent_sigmoid + latent * latent_sigmoid
        pred_mid= self.forward_decoder_mid(mid_latent,ids_restore)

        loss_mid = self.forward_loss_r(resnet_latent_mid, pred_mid, mask)
        loss_last = self.forward_loss_r(resnet_latent_last, pred_last, mask)

        loss_mid , loss_last = loss_mid.mean() , loss_last.mean()
        loss = loss_mid + loss_last
        return  loss , loss_mid , loss_last

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks