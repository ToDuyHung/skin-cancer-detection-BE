import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EffNetMetaImageAttentionV3(nn.Module):
    def __init__(self, base_model, meta_attn_blk_cls, attention_indices, num_classes, d_meta, dropout=0.1, embed_dim=512):
        super().__init__()
        self.meta_attn_blk_cls = meta_attn_blk_cls
        self.d_meta = d_meta
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.num_classes = num_classes
 
        self.ftr_extractors = nn.ModuleList()
        old_idx = 0
        for idx in attention_indices:
            self.ftr_extractors.append(base_model[old_idx:idx+1])
            old_idx = idx+1
        if old_idx < len(base_model):
            self.ftr_extractors.append(base_model[old_idx:])
        self.num_attn = len(attention_indices)
        self.attn_blks, self.out_channels = self._attn_blocks()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.out_channels, num_classes)
        )
 
        self._init_xavierUniform(self.attn_blks)
        self._init_xavierUniform(self.classifier)
    
    def forward(self, X, get_attn_maps=False):
        img, meta = X
        y = img
        out_ctx = []
        out_attn = []
        for i in range(self.num_attn):
            y = self.ftr_extractors[i](y)
            out_ctx.append(y)
        if self.num_attn < len(self.ftr_extractors):
            y = self.ftr_extractors[-1](y)
        g = self.avgpool(y).flatten(1)
        for i in range(self.num_attn):
            ctx, attn = self.attn_blks[i](out_ctx[i], meta, g)
            out_ctx[i] = ctx
            out_attn.append(attn)
        g = torch.cat(out_ctx, dim=1)
        y = self.classifier(g)
        if get_attn_maps:
            return y, out_attn
        return y
 
    def _attn_blocks(self):
        attn_blks = nn.ModuleList()
        out = torch.zeros(1, 3, 1, 1)
        out_channels = []
        self.eval()
        with torch.no_grad():
            for i in range(self.num_attn):
                out = self.ftr_extractors[i](out)
                out_channels.append(out.shape[1])
            if self.num_attn < len(self.ftr_extractors):
                out = self.ftr_extractors[-1](out)
        self.train()
        for ch in out_channels:
            attn_blks.append(
                # self.MetaImageAttentionBlock(self.d_meta,
                #                              ch,
                #                              out_channels[-1],
                #                              self.embed_dim,
                #                              self.dropout)
                self.meta_attn_blk_cls(self.d_meta,
                                       ch,
                                       out.shape[1],
                                       self.embed_dim,
                                       self.dropout)
            )
        return attn_blks, sum(out_channels)
    
    def unfreeze_img_conv(self):
        for param in self.ftr_extractors.parameters():
            param.requires_grad = True
    
    def freeze_img_conv(self):
        for param in self.ftr_extractors.parameters():
            param.requires_grad = False
 
    def _init_xavierUniform(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain = np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, a = 0, b = 1)
                nn.init.constant_(m.bias, val = 0.)
    
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain = np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, val = 0.)