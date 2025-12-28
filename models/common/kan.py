import torch
import torch.nn as nn

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        # 权重矩阵
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # 可调节的 B-spline 或其他网格
        self.grid = nn.Parameter(torch.randn(grid_size, in_features))  # grid_size 是 B-spline 网格的大小

    def forward(self, x):
        # 假设 x 形状为 (BT, HW+1, D)，其中 D 是输入的特征维度
        
        # 计算每个输入样本与每个网格的交互
        interaction = torch.matmul(x, self.grid.t())  # interaction 形状应该是 (BT, HW+1, grid_size)
        
        # 使用 Sigmoid 激活，模拟非线性转换
        non_linear_map = torch.sigmoid(interaction)   # 形状 (BT, HW+1, grid_size)
        
        # 这里确保非线性映射的维度正确，并进行矩阵乘法
        # 对 non_linear_map 执行 reshape 操作，使其成为 (BT * HW+1, grid_size)
        non_linear_map = non_linear_map.view(-1, self.grid_size)
        
        # 确保矩阵乘法维度匹配
        output = torch.matmul(non_linear_map, self.weight.t()) + self.bias
        
        # 返回输出
        return output
