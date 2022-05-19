import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class AdditiveMetaAttentionBlock(nn.Module):
    def __init__(self, d_meta, c_img, d_glob, embed_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim if embed_dim else c_img
        self.d_meta = d_meta
        self.meta_embed = nn.Sequential(
            nn.Linear(d_meta, self.embed_dim, bias=False),
            # nn.GELU()
        )

        self.img_embed = nn.Sequential(
            nn.Conv2d(in_channels=c_img, out_channels=c_img, kernel_size=3, padding='same', groups=c_img, bias=False),
            nn.Conv2d(in_channels=c_img, out_channels=embed_dim, kernel_size=1, bias=False)
            # nn.Conv2d(c_img, embed_dim, kernel_size=3, padding='same', bias=False),
            # nn.Conv2d(c_img, embed_dim, kernel_size=7, padding='same'),
            # nn.GELU()
        ) if embed_dim else nn.Identity()

        self.glob_embed = nn.Linear(d_glob, self.embed_dim, bias=False)

        # self.score = nn.Conv2d(embed_dim, 1, kernel_size=3, padding='same', bias=False)
        self.score = nn.Linear(embed_dim, 1, bias=False)

        self.norm = nn.LayerNorm(c_img)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X_img, X_meta, global_ftr):
        # X_img: N x c x h x w
        # X_meta: N x d_meta
        N, _, h, w = X_img.size()

        meta_embed = self.dropout(self.meta_embed(X_meta)) # N x embed_dim
        img_embed = self.img_embed(X_img) # N x embed_dim x h x w
        g_embed = self.dropout(self.glob_embed(global_ftr)) # N x embed_dim

        img_embed = img_embed.flatten(2) # N x embed_dim x h*w
        # Scaled Dot Production
        c = img_embed + meta_embed.unsqueeze(-1) + g_embed.unsqueeze(-1) # N x embed_dim x h*w
        # score = self.score(torch.tanh(c.view(N, -1, h, w))) # N x 1 x h x w
        score = self.score(torch.tanh(c.transpose(1, 2))) # N x h*w x 1
        attn_map = F.softmax(score.view(N, 1, -1), dim=2) # N x 1 x h*w
        
        # N x 1 x c
        context = torch.matmul(attn_map, X_img.flatten(-2).transpose(-2, -1))
        context = self.norm(context.squeeze(1))
        attn_map = attn_map.view(N, 1, h, w)
        # context = (1 + attn_map) * X_img
        # (N x c x h x w, N x h x w)
        return context, attn_map.squeeze(1)