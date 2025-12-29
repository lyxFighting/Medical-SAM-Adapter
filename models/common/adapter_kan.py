import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, skip_connect=True, grid_size=5):
        super().__init__()
        self.skip_connect = skip_connect

        D_hidden_features = int(D_features * mlp_ratio)

        self.D_fc1 = nn.Linear(D_features, D_hidden_features)

        # 用 KANLayer 替代 act
        self.act = KANActivation(D_hidden_features, grid_size=grid_size) 

        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x: (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)     # ← 这里不再是 GELU
        xs = self.D_fc2(xs)

        if self.skip_connect:
            return x + xs
        else:
            return xs



class KANActivation(nn.Module):
    """
    KAN as a learnable activation function (Safe Version)
    """

    def __init__(
        self,
        channels,
        grid_size=5,
        spline_order=3,
        scale_base=1.0,
        scale_spline=1.0,
    ):
        super().__init__()

        self.channels = channels
        self.grid_size = grid_size
        self.spline_order = spline_order

        # ---------- Base activation ----------
        self.base_act = nn.SiLU()
        self.scale_base = scale_base

        # ---------- Input normalization ----------
        self.norm = nn.LayerNorm(channels)

        # ---------- Spline parameters ----------
        num_basis = grid_size + spline_order
        self.spline_weight = nn.Parameter(
            torch.zeros(channels, num_basis)
        )

        self.scale_spline = scale_spline

        # ---------- Fixed uniform grid ----------
        h = 2.0 / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1)
            * h - 1.0
        )
        self.register_buffer("grid", grid)

    def forward(self, x):
        """
        x: [B, ..., C]
        """
        shape = x.shape
        x = x.view(-1, self.channels)

        # Normalize for numerical stability
        x_norm = self.norm(x)

        # Base activation
        base_out = self.base_act(x_norm) * self.scale_base

        # Clamp only after normalization
        x_clamped = torch.clamp(x_norm, -1.0, 1.0)

        # B-spline basis
        basis = self._b_spline_basis(x_clamped)
        # [B, C, num_basis]

        spline_out = torch.sum(
            basis * self.spline_weight.unsqueeze(0),
            dim=-1
        ) * self.scale_spline

        out = base_out + spline_out
        return out.view(*shape)

    def _b_spline_basis(self, x):
        """
        Cox-De Boor recursion (vectorized)
        x: [B, C]
        return: [B, C, grid_size + spline_order]
        """
        grid = self.grid
        x = x.unsqueeze(-1)  # [B, C, 1]
        k = self.spline_order
        eps = 1e-8  # 防止除零

        # 0th-order basis
        # bases[i] = 1 if grid[i] <= x < grid[i+1], else 0
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()

        # Recursive computation for higher-order B-splines
        for p in range(1, k + 1):
            # 获取前一个阶数的基函数（去掉最后一个和第一个）
            left = bases[:, :, :-1]  # B_{i, p-1}
            right = bases[:, :, 1:]   # B_{i+1, p-1}

            # 计算分母，添加 epsilon 防止除零
            # denom1 = t_{i+p} - t_i
            denom1 = grid[p:-1] - grid[:-(p + 1)]
            denom1 = denom1 + eps * (denom1 == 0).float()
            
            # denom2 = t_{i+p+1} - t_{i+1}
            denom2 = grid[p + 1:] - grid[1:-p]
            denom2 = denom2 + eps * (denom2 == 0).float()

            # Cox-De Boor 递归公式
            # B_{i,p}(x) = (x - t_i)/(t_{i+p} - t_i) * B_{i,p-1}(x) 
            #            + (t_{i+p+1} - x)/(t_{i+p+1} - t_{i+1}) * B_{i+1,p-1}(x)
            term1 = (x - grid[:-(p + 1)]) / denom1 * left
            term2 = (grid[p + 1:] - x) / denom2 * right

            bases = term1 + term2

        return bases
