import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# 多尺度因果卷积 TCN 模块
# ================================================================
class TCNBlock(nn.Module):
    """
    多尺度 TCN 模块：
    - 每层包含多个不同卷积核大小的卷积分支（如3、5、7）
    - 使用因果填充，确保时间因果性
    - 每层包含 LayerNorm、ReLU、Dropout 以及残差连接
    """
    def __init__(self, input_dim, hidden_dim, output_dim=None, num_layers=2,
                 kernel_size=[3, 5, 7], dropout=0.1):
        super(TCNBlock, self).__init__()
        self.num_layers = num_layers
        self.dilations = []

        self.multi_scale_convs = nn.ModuleList()
        self.norms, self.relus, self.dropouts = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim * len(kernel_size)
            out_channels = hidden_dim
            dilation = 2 ** i
            self.dilations.append(dilation)

            # 多尺度卷积分支
            scale_convs = nn.ModuleList([
                nn.Conv1d(in_channels, out_channels, k, padding=0, dilation=dilation)
                for k in kernel_size
            ])
            self.multi_scale_convs.append(scale_convs)

            # 标准化与激活层
            self.norms.append(nn.LayerNorm(hidden_dim * len(kernel_size)))
            self.relus.append(nn.ReLU())
            self.dropouts.append(nn.Dropout(dropout))

        # 残差匹配
        self.residual_proj = (
            nn.Conv1d(input_dim, hidden_dim * len(kernel_size), kernel_size=1)
            if input_dim != hidden_dim * len(kernel_size) else None
        )

    def forward(self, x):
        """
        输入: [batch, seq_len, input_dim]
        输出: [batch, seq_len, hidden_dim * len(kernel_size)]
        """
        x = x.transpose(1, 2)
        residual = x

        for i, (scale_convs, norm, relu, dropout, dilation) in enumerate(
                zip(self.multi_scale_convs, self.norms, self.relus, self.dropouts, self.dilations)):

            # 多尺度卷积
            scale_outputs = []
            for conv in scale_convs:
                pad = (dilation * (conv.kernel_size[0] - 1), 0)
                scale_x = F.pad(x, pad)
                scale_outputs.append(conv(scale_x))

            # 拼接特征
            x = torch.cat(scale_outputs, dim=1)

            # 归一化 + 激活 + Dropout
            x = x.transpose(1, 2)
            x = norm(x)
            x = x.transpose(1, 2)
            x = relu(x)
            x = dropout(x)

            # 残差连接
            if i == 0 and self.residual_proj is not None:
                residual = self.residual_proj(residual)
            x = x + residual
            residual = x

        return x.transpose(1, 2)


# ================================================================
# 相对位置编码
# ================================================================
class RelativePositionBias(nn.Module):
    def __init__(self, max_len, n_heads=None):
        super().__init__()
        self.max_len = max_len
        # 共享头：只要 1 维标量；如果你想每个头一套，可设 n_heads 并用 Embedding(..., n_heads)
        self.rel_bias = nn.Embedding(2 * max_len - 1, 1)

    def forward(self, seq_len, device):
        i = torch.arange(seq_len, device=device)
        j = torch.arange(seq_len, device=device)
        rel = (i[:, None] - j[None, :]).clamp(-(self.max_len - 1), self.max_len - 1)
        idx = rel + (self.max_len - 1)         # [L, L] in [0, 2*max_len-2]
        bias = self.rel_bias(idx).squeeze(-1)  # [L, L]  标量偏置
        bias = torch.clamp(bias, -5.0, 5.0)  # 限幅，防软最大溢出
        bias = torch.nan_to_num(bias, nan=0.0, neginf=-5.0, posinf=5.0)
        return bias

# ================================================================
# 主模型：TCN + Transformer 并行融合结构
# ================================================================
class AnomalyDetectionModel(nn.Module):
    """
    异常检测模型：
    - TCN 提取局部时序特征
    - Transformer 捕获全局依赖
    - 双向交叉注意力融合 + 门控机制
    """
    def __init__(self, feature_dim, cross_attn_layers=2, model_dim=128, num_classes=5,
                 tcn_layers=2, transformer_layers=2, nheads=8,
                 dropout=0.1, max_len=500):
        super(AnomalyDetectionModel, self).__init__()

        # 输入映射层
        self.feature_proj = nn.Linear(feature_dim, model_dim) if feature_dim != model_dim else None
        self.tcn_output_dim = model_dim * 3

        # 模块定义
        self.tcn = TCNBlock(model_dim, model_dim, num_layers=tcn_layers,
                            kernel_size=[3, 5, 7], dropout=dropout)
        self.rel_pos_bias = RelativePositionBias(max_len)

        encoder_layer = None
        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=nheads,
                dropout=dropout,
                batch_first=True,
                enable_nested_tensor=False,  # 新版支持
            )
        except TypeError:
            # 老版没有 enable_nested_tensor 参数，退回不带该参数的写法
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=nheads,
                dropout=dropout,
                batch_first=True,
            )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # 双向交叉注意力
        self.cross_attn_layers = cross_attn_layers
        self.tcn_proj_to_256 = nn.Linear(self.tcn_output_dim, 256)
        self.trans_proj_to_256 = nn.Linear(model_dim, 256)

        self.tcn_to_trans_attns = nn.ModuleList([
            nn.MultiheadAttention(256, nheads, batch_first=True)
            for _ in range(self.cross_attn_layers)
        ])
        self.trans_to_tcn_attns = nn.ModuleList([
            nn.MultiheadAttention(256, nheads, batch_first=True)
            for _ in range(self.cross_attn_layers)
        ])

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.Sigmoid()
        )
        self.fusion_layer = nn.Linear(256, model_dim)

        # 注意力池化
        self.attention_pool = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1)
        )

        # 分类层
        self.fc = nn.Linear(model_dim, num_classes)

        # 卷积层权重初始化（恢复你原模型的特性）
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x, pad_mask: torch.Tensor | None = None,return_features: bool = False):
        """前向传播流程"""
        # 1. 特征映射
        if self.feature_proj is not None:
            x = self.feature_proj(x)

        # 2. TCN 分支
        x_tcn = self.tcn(x)

        # 3. Transformer 分支（稳定写法 ✅）
        seq_len = x.size(1)
        # 构造 “加性注意力偏置” [L, L]，注入到 TransformerEncoder
        attn_bias = self.rel_pos_bias(seq_len, x.device)  # [L, L], float32
        attn_bias = torch.clamp(attn_bias, -5.0, 5.0)
        attn_bias = torch.nan_to_num(attn_bias, nan=0.0, neginf=-5.0, posinf=5.0)

        # 关键：把 pad_mask 作为 key padding mask 传入 Transformer
        x_trans = self.transformer(
            x,
            src_key_padding_mask=pad_mask  # 新增：屏蔽掉 padding 位置
        )


        # 4. 投影到 256 维
        x_tcn_proj = self.tcn_proj_to_256(x_tcn)
        x_trans_proj = self.trans_proj_to_256(x_trans)

        # 5. 多层交叉注意力（每一层都传入 attn_mask）
        tcn_att, trans_att = x_tcn_proj, x_trans_proj
        for t2t, t2c in zip(self.tcn_to_trans_attns, self.trans_to_tcn_attns):
            # TCN <- Trans
            tcn_att, _ = t2t(
                tcn_att,  # query
                trans_att,  # key
                trans_att,  # value
                key_padding_mask=pad_mask
            )
            # Trans <- TCN
            trans_att, _ = t2c(
                trans_att,  # query
                tcn_att,  # key
                tcn_att,  # value
                key_padding_mask=pad_mask
            )
        # 6. 门控融合
        gate_weight = self.gate(torch.cat([tcn_att, trans_att], dim=-1))
        fused_output = gate_weight * tcn_att + (1 - gate_weight) * trans_att
        fused_output = self.fusion_layer(fused_output)

        # 7. 注意力池化
        attn_weights = F.softmax(self.attention_pool(fused_output), dim=1)
        seq_repr = (fused_output * attn_weights).sum(dim=1)

        # 8. 分类输出
        logits = self.fc(seq_repr)
        if return_features:
            return logits, seq_repr
        return logits


