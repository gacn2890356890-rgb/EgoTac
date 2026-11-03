"""Train the complete spatial/region/temporal tactile decoder on frozen features."""
import torch
from torch import nn
from torch.nn import functional as F


class JointTactileHead(nn.Module):
    def __init__(self, prior, positive_median, mode, width=128, context=40):
        super().__init__()
        assert mode in ('global', 'hand')
        self.mode, self.context = mode, context
        self.visual = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, width))
        self.motion = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, width, bias=False))
        self.position = nn.Linear(2, width, bias=False)
        self.geometry = nn.Sequential(nn.Linear(28, width), nn.GELU(), nn.Linear(width, width))
        self.region_query = nn.Embedding(12, width)
        self.cross = nn.ModuleList([nn.MultiheadAttention(width, 4, dropout=.1, batch_first=True) for _ in range(2)])
        self.cross_norm = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        def encoder(layers):
            block = nn.TransformerEncoderLayer(width, 4, 4*width, dropout=.1,
                batch_first=True, norm_first=True, activation='gelu')
            return nn.TransformerEncoder(block, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.region = encoder(1)
        self.temporal = encoder(2)
        self.time = nn.Parameter(torch.zeros(1, context, width))
        self.contact = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        self.amplitude = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        prior = torch.as_tensor(prior).clamp(.001, .999)
        median = torch.as_tensor(positive_median).clamp(min=.1)
        self.register_buffer('prior_logit', torch.logit(prior))
        self.register_buffer('amplitude_offset', torch.log(torch.expm1(median)))
        self.register_buffer('grid_uv', torch.tensor([[(x+.5)/8, (y+.5)/8] for y in range(8) for x in range(8)]))
        nn.init.normal_(self.time, std=.01)
        nn.init.zeros_(self.contact[-1].weight)
        nn.init.zeros_(self.contact[-1].bias)
        nn.init.zeros_(self.amplitude[-1].weight)
        nn.init.zeros_(self.amplitude[-1].bias)

    def forward(self, grid, geometry, uv, geometry_valid, local_grid, local_uv, local_valid):
        b, t = grid.shape[:2]
        assert grid.shape[-2:] == (64, 1024) and t <= self.context
        coords = self.grid_uv.view(1, 1, 64, 2).expand(b, t, -1, -1)
        valid = torch.ones(b, t, 64, dtype=torch.bool, device=grid.device)
        x = grid.float()
        if self.mode == 'hand':
            x = torch.cat((x, local_grid.float()), 2)
            coords = torch.cat((coords, local_uv), 2)
            valid = torch.cat((valid, local_valid.bool()), 2)
        dx = torch.cat((torch.zeros_like(x[:, :1]), x[:, 1:]-x[:, :-1]), 1)
        tokens = self.visual(x) + self.motion(dx) + self.position(coords)
        tokens = F.dropout(tokens, .1, self.training)
        dg = torch.cat((torch.zeros_like(geometry[:, :1]), geometry[:, 1:]-geometry[:, :-1]), 1)
        query = self.region_query.weight.view(1, 1, 12, -1).expand(b, t, -1, -1)
        query = query + self.geometry(torch.cat((geometry, dg), -1))*geometry_valid[..., None]
        distance = ((uv[..., None, :]-coords[:, :, None])**2).sum(-1)
        bias = -distance/(2*.25**2)*geometry_valid[..., None]
        bias = bias.masked_fill(~valid[:, :, None], float('-inf'))
        if self.mode == 'hand':
            bias[:, :, :6, 80:96] = float('-inf')
            bias[:, :, 6:, 64:80] = float('-inf')
        n, w = x.shape[2], query.shape[-1]
        bias = bias.reshape(b*t, 1, 12, n).expand(-1, 4, -1, -1).reshape(b*t*4, 12, n)
        z = query.reshape(b*t, 12, w)
        tokens = tokens.reshape(b*t, n, w)
        for attention, norm in zip(self.cross, self.cross_norm):
            fused, _ = attention(norm(z), tokens, tokens, attn_mask=bias, need_weights=False)
            z = z + fused
        z = self.region(z).reshape(b, t, 12, w).permute(0, 2, 1, 3).reshape(b*12, t, w)
        z = self.temporal(z + self.time[:, :t]).reshape(b, 12, t, w).permute(0, 2, 1, 3)
        logits = self.contact(z).squeeze(-1) + self.prior_logit
        amplitude_log = F.softplus(self.amplitude(z).squeeze(-1)+self.amplitude_offset)
        amplitude = torch.expm1(amplitude_log.clamp(max=6))
        return {'contact_logits': logits, 'amplitude_log': amplitude_log,
                'conditional_response': amplitude, 'response': logits.sigmoid()*amplitude}
