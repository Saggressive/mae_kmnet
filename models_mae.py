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
import torch.nn.functional as F
from util.pos_embed import get_2d_sincos_pos_embed
from resnet import resnet101,resnet50



class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 momentum=0.996,queue_size=65536,temp=0.07):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_m = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])

        self.blocks_m = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.norm_m = norm_layer(embed_dim)

        self.middle_norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed_features = nn.Linear(embed_dim, embed_dim, bias=True)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_cder = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        # self.decoder_embed_feat = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_cder = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_features = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed_feat = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(4)])

        self.decoder_blocks_cder = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(4)])

        self.decoder_blocks_features = nn.ModuleList([
            Block(embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(4)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim,1024, bias=True) # decoder to patch
        self.decoder_norm_cder = norm_layer(decoder_embed_dim)
        self.decoder_pred_cder = nn.Linear(decoder_embed_dim, 1024, bias=True)
        self.decoder_norm_features = norm_layer(embed_dim)

        self.norm_pix_loss = norm_pix_loss
        self.resnet50=resnet50(pretrained=True)
        self.initialize_weights()
        # --------------------------------------------------------------------------
        self.prejector=self._build_mlp(3,768,4096,256)
        self.prejector_m=self._build_mlp(3,768,4096,256)
        self.prediction=self._build_mlp(2,256,4096,256)

        self.momentum = momentum
        self.model_pairs = [[self.blocks,self.blocks_m],
                            [self.norm,self.norm_m],
                            [self.prejector,self.prejector_m]
                           ]
        self.copy_params()
        # create the queue
        self.queue_size=queue_size
        self.register_buffer("queue", torch.randn(256,self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.temp = temp


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        decoder_pos_embed_feat = get_2d_sincos_pos_embed(self.decoder_pos_embed_feat.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed_feat.data.copy_(torch.from_numpy(decoder_pos_embed_feat).float().unsqueeze(0))
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.cls_token_m, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.mask_token_cder, std=.02)
        torch.nn.init.normal_(self.mask_token_features, std=.02)

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

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, feat):
        # gather keys before updating queue
        feats = concat_all_gather(feat)
        # feats=feat
        batch_size = feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr 

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

    def forward_encoder_m(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk_i,blk in enumerate(self.blocks_m):
            x = blk(x)

        x = self.norm_m(x)
        x_copy = x.clone().detach()
        x_mean = x_copy[:, 1:].mean(dim=1)
        x_pjc = self.prejector_m(x_mean)
        return x_pjc , x

    def forward_decoder_t(self,x):
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_decoder_r(self,x):
        # apply Transformer blocks
        for blk in self.decoder_blocks_cder:
            x = blk(x)
        x = self.decoder_norm_cder(x)

        # predictor projection
        x = self.decoder_pred_cder(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_decoder_features(self,x):
        # apply Transformer blocks
        for blk in self.decoder_blocks_features:
            x = blk(x)
        x = self.decoder_norm_features(x)

        x_copy=x.clone()
        x_mean = x_copy[:, 1:].mean(dim=1)
        x_mean=self.prejector(x_mean)
        x_mean=self.prediction(x_mean)

        return x_mean ,x

    def forward_decoder(self, x, middle_x,ids_restore):
        # embed tokens
        feat_x=x.clone()
        feat_x=self.decoder_embed_features(x)
        x = self.decoder_embed(x)
        middle_x = self.decoder_embed_cder(middle_x)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        mask_tokens_cder = self.mask_token_cder.repeat(middle_x.shape[0], ids_restore.shape[1] + 1 - middle_x.shape[1], 1)
        middle_x_ = torch.cat([middle_x[:, 1:, :], mask_tokens_cder], dim=1)  # no cls token
        middle_x_ = torch.gather(middle_x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, middle_x.shape[2]))  # unshuffle
        middle_x = torch.cat([middle_x[:, :1, :], middle_x_], dim=1)  # append cls token

        mask_tokens_feat = self.mask_token_features.repeat(feat_x.shape[0], ids_restore.shape[1] + 1 - feat_x.shape[1], 1)
        feat_x_ = torch.cat([feat_x[:, 1:, :], mask_tokens_feat], dim=1)  # no cls token
        feat_x_ = torch.gather(feat_x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, feat_x.shape[2]))  # unshuffle
        feat_x = torch.cat([feat_x[:, :1, :], feat_x_], dim=1)  # append cls token
        # add pos embed
        x = x + self.decoder_pos_embed
        middle_x = middle_x + self.decoder_pos_embed
        feat_x = feat_x + self.decoder_pos_embed_feat

        t=self.forward_decoder_t(middle_x)
        r=self.forward_decoder_r(x)
        m,feat=self.forward_decoder_features(feat_x)

        return t,r,m,feat



    def forward_loss_t_r(self, target, pred1, pred2,mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        n,l,d=target.size()
        target=target.reshape(n,-1)
        mean=target.mean(dim=0,keepdim=True)
        var = target.var(dim=0, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5
        target=target.reshape(n,l,d)

        loss1 = (pred1 - target) ** 2
        loss1 = loss1.mean(dim=-1)  # [N, L], mean loss per patch
        loss1 = (loss1 * mask).sum() / mask.sum()  # mean loss on removed patches

        loss2 = (pred2 - target) ** 2
        loss2 = loss2.mean(dim=-1)  # [N, L], mean loss per patch
        loss2 = (loss2 * mask).sum() / mask.sum()  # mean loss on removed patches

        return loss1,loss2

    def forward_loss_feat(self, pred, target):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """

        loss = (pred - target) ** 2
        loss = loss.mean()  # [N, L], mean loss per patch
        return loss

    def contrastive_loss(self, q, k):
        with torch.no_grad():
            q=F.normalize(q,dim=1) 
            feats = torch.cat([q.t(),self.queue.clone().detach()],dim=1)                                         

            sim_m = q @ feats / self.temp     

            sim_targets = torch.zeros(sim_m.size()).to(q.device)
            sim_targets.fill_diagonal_(1)          
        k=F.normalize(k,dim=1)
        sim = k @ feats / self.temp 

        cl_loss = -torch.sum(F.log_softmax(sim, dim=1)*sim_targets,dim=1).mean()
        self._dequeue_and_enqueue(q)

        return cl_loss

    def MAE(self,imgs, mask_ratio, resnet_latent):
        latent,m_latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred_t,pred_r,k,feat = self.forward_decoder(latent,m_latent, ids_restore)  # [N, L, p*p*3]
        t_loss,r_loss = self.forward_loss_t_r(resnet_latent, pred_r,pred_t, mask)
        return t_loss,r_loss,k,feat


    def forward(self, imgs0, imgs1,mask_ratio=0.75):
        with torch.no_grad():
            resnet_latent=self.resnet50(imgs0)
            resnet_latent=resnet_latent.detach()
            resnet_latent=resnet_latent.permute(0,2,3,1)
            n,w,h,d=resnet_latent.size()
            resnet_latent=resnet_latent.reshape(shape=(-1,w*h,d))
        t_loss,r_loss,k,imgs0_feat = self.MAE(imgs0, mask_ratio, resnet_latent)

        with torch.no_grad():
            self._momentum_update()
            q,_=self.forward_encoder_m(imgs1)
            _,imgs0_feat_m=self.forward_encoder_m(imgs0)
        cl_loss=self.contrastive_loss(q, k)
        re_loss=self.forward_loss_feat(imgs0_feat, imgs0_feat_m)
        loss = t_loss + r_loss + cl_loss + re_loss*0.1
        return loss, t_loss , r_loss ,cl_loss, re_loss


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

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks