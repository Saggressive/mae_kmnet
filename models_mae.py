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
import clip



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
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_m = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_m = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(6)])

        self.decoder_blocks_clip = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(2)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        self.decoder_norm_clip = norm_layer(decoder_embed_dim)
        self.decoder_pred_clip = nn.Linear(decoder_embed_dim, 768, bias=True)
        # --------------------------------------------------------------------------
    
        self.norm_pix_loss = norm_pix_loss
        self.clip, self.preprocess = clip.load("ViT-B/16")
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
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.mask_token_m, std=.02)

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
                h_early=x[:,1:,:].clone()

        cls_late=x[:,:1,:].clone()
        middle_x=torch.cat([cls_late,h_early],dim=1)
        x = self.norm(x)
        middle_x = self.middle_norm(middle_x)
        return x, middle_x,mask, ids_restore

        # for blk_i,blk in enumerate(self.blocks):
        #     x = blk(x)

        # x = self.norm(x)
        # return x, x.clone(),mask, ids_restore


    def forward_decoder_pixel(self,x):
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_decoder_clip(self,x):
        # apply Transformer blocks
        for blk in self.decoder_blocks_clip:
            x = blk(x)
        x = self.decoder_norm_clip(x)

        # predictor projection
        x = self.decoder_pred_clip(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_decoder_clip_main(self, x,ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        clip_pred=self.forward_decoder_clip(x)

        return clip_pred

    def forward_decoder(self, x, m_x,ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        m_x = self.decoder_embed_m(m_x)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        m_mask_tokens = self.mask_token_m.repeat(m_x.shape[0], ids_restore.shape[1] + 1 - m_x.shape[1], 1)
        m_x_ = torch.cat([m_x[:, 1:, :], m_mask_tokens], dim=1)  # no cls token
        m_x_ = torch.gather(m_x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, m_x.shape[2]))  # unshuffle
        m_x = torch.cat([m_x[:, :1, :], m_x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed
        m_x = m_x + self.decoder_pos_embed

        clip_pred=self.forward_decoder_clip(x)
        pixel_pred=self.forward_decoder_pixel(m_x)

        return clip_pred,pixel_pred

    def forward_loss_clip(self, clip_target, clip_pred,mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """

        clip_target=clip_target[:,1:,:]
        clip_mean=clip_target.mean(dim=-1,keepdim=True)
        clip_var = clip_target.var(dim=-1, keepdim=True)
        clip_target = (clip_target - clip_mean) / (clip_var + 1.e-6)**.5

        clip_loss = (clip_target - clip_pred) ** 2
        clip_loss = clip_loss.mean(dim=-1)  # [N, L], mean loss per patch
        clip_loss = (clip_loss * mask).sum() / mask.sum()  # mean loss on removed patches


        return clip_loss

    def forward_loss(self, clip_target,pixel_target, clip_pred, pixel_pred,mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        pixel_target = self.patchify(pixel_target)
        pixel_mean = pixel_target.mean(dim=-1, keepdim=True)
        pixel_var = pixel_target.var(dim=-1, keepdim=True)
        pixel_target = (pixel_target - pixel_mean) / (pixel_var + 1.e-6)**.5

        clip_target=clip_target[:,1:,:]
        clip_mean=clip_target.mean(dim=-1,keepdim=True)
        clip_var = clip_target.var(dim=-1, keepdim=True)
        clip_target = (clip_target - clip_mean) / (clip_var + 1.e-6)**.5

        clip_loss = (clip_target - clip_pred) ** 2
        clip_loss = clip_loss.mean(dim=-1)  # [N, L], mean loss per patch
        clip_loss = (clip_loss * mask).sum() / mask.sum()  # mean loss on removed patches

        pixel_loss = (pixel_target - pixel_pred) ** 2
        pixel_loss = pixel_loss.mean(dim=-1)  # [N, L], mean loss per patch
        pixel_loss = (pixel_loss* mask).sum() / mask.sum()  # mean loss on removed patches

        return clip_loss , pixel_loss

    def forward(self, imgs, mask_ratio=0.75):
        with torch.no_grad():
            _,clip_latent=self.clip.encode_image(imgs)
            clip_latent=clip_latent.detach()

        latent,m_latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        clip_pred,pixel_pred = self.forward_decoder(latent,m_latent, ids_restore)  # [N, L, p*p*3]
        clip_loss,pixel_loss = self.forward_loss(clip_latent, imgs,clip_pred,pixel_pred, mask)
        loss = clip_loss + 0.5*pixel_loss
        return loss, clip_loss , pixel_loss

        # clip_pred= self.forward_decoder_clip_main(latent,ids_restore)  # [N, L, p*p*3]
        # clip_loss= self.forward_loss_clip(clip_latent, clip_pred, mask)
        # loss = clip_loss 
        # return loss, clip_loss , torch.tensor(0,dtype=torch.float16)


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