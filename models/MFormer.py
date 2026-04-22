"""
MFormer
"""

from timm.layers import DropPath
from utils.position_coding import LearnablePositionalEmbedding, SinusoidalPositionalEncoding, RotaryPositionalEncoding
import torch
import torch.nn as nn
import torch.nn.functional as F


class MinimalLRU(nn.Module):
    """
    基于并行扫描 (Parallel Scan/FFT Convolution) 的极简 1D 线性递归单元 (LRU)。
    利用对角化复数矩阵实现，可高效处理超长序列。
    (已将复数参数拆分为实部和虚部，完美兼容 PyTorch AMP 混合精度训练)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.nu = nn.Parameter(torch.randn(dim))
        self.theta = nn.Parameter(torch.randn(dim))

        # 拆分复数投影参数，避免 GradScaler 处理 ComplexFloat 时崩溃
        self.gamma_real = nn.Parameter(torch.randn(dim))
        self.gamma_imag = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # 1. 计算复数指数参数: lambda = exp(-exp(nu) + i*theta)
        lambdas = torch.exp(-torch.exp(self.nu) + 1j * self.theta)  # [d]

        # 2. 构建时间步
        time_steps = torch.arange(L, device=x.device)  # [L]
        lambda_powers = lambdas.unsqueeze(0) ** time_steps.unsqueeze(1)  # [L, d]

        # 在计算图中动态重构复数 gamma
        gamma = torch.complex(self.gamma_real, self.gamma_imag)
        kernel = gamma * lambda_powers  # [L, d]

        # 3. 使用全量 FFT 进行高效并行卷积
        kernel_f = torch.fft.fft(kernel, n=2 * L, dim=0)  # [2L, d]
        x_f = torch.fft.fft(x.to(torch.cfloat), n=2 * L, dim=1)  # [B, 2L, d]

        # 频域相乘后，使用 IFFT 变换回时域
        y_f = x_f * kernel_f.unsqueeze(0)
        y = torch.fft.ifft(y_f, n=2 * L, dim=1)  # [B, 2L, d]

        # 截取真实的序列长度并提取实部
        return y[:, :L, :].real


class BiLRU(nn.Module):
    """双向 LRU 动力学处理模块。"""

    def __init__(self, dim: int):
        super().__init__()
        self.fwd_lru = MinimalLRU(dim)
        self.bwd_lru = MinimalLRU(dim)
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 组合输入序列，形状为 [B, L, d]
        Returns:
            torch.Tensor: 演化后的序列，形状为 [B, L, d]
        """
        # 前向动力学演化
        y_fwd = self.fwd_lru(x)
        # 后向动力学演化：将输入按时间步翻转 -> 演化 -> 再翻转复原
        y_bwd = self.bwd_lru(x.flip(dims=[1])).flip(dims=[1])

        # 拼接前后向结果，通过线性层降维至原特征维度
        return self.proj(torch.cat([y_fwd, y_bwd], dim=-1))


class MemoryTimeUnit(nn.Module):
    def __init__(
            self,
            dim: int,
    ):
        super().__init__()

        # 可学习的模态类型参数
        self.prefix_type_emb = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.signal_type_emb = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # 全局时序交互的 BiLRU 算子
        self.bilru = BiLRU(dim)

    def forward(self, memory: torch.Tensor, ts_embeds: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:

        Batch, P, d = memory.size()
        _, L_p, V, _ = ts_embeds.size()

        # 步骤 1: 通道独立的数据重构 (Channel-Independent Reshape)
        ts_embeds_reshaped = ts_embeds.transpose(1, 2).contiguous().view(Batch * V, L_p, d)
        memory_expanded = memory.unsqueeze(1).expand(Batch, V, P, d).contiguous().view(Batch * V, P, d)

        # 广播相加模态类型参数，显式区分指令区与数据区
        memory_encoded = memory_expanded + self.prefix_type_emb  # [Batch * V, P, d]
        ts_embeds_encoded = ts_embeds_reshaped + self.signal_type_emb  # [Batch * V, L_p, d]

        # 拼接序列，得到组合序列 Z
        Z = torch.cat([memory_encoded, ts_embeds_encoded], dim=1)  # [Batch * V, P + L_p, d]

        # 步骤 2: 双向动力学演化 (BiLRU Processing)
        Z_prime = self.bilru(Z)  # 形状保持为 [Batch * V, P + L_p, d]

        # 步骤 3: 瓶颈特征提取与通道聚合 (Extraction & Aggregation)
        memory_prime = Z_prime[:, :P, :]  # 截取更新后的记忆片段 [Batch * V, P, d]
        memory_prime = memory_prime.view(Batch, V, P, d)  # 还原形状 [Batch, V, P, d]

        # 变量/通道维度进行 mean pooling
        output = memory_prime.mean(dim=1)  # [Batch, P, d]

        return output

class MTUBlock(nn.Module):

    def __init__(
            self,
            dim,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.mut = MemoryTimeUnit(
            dim
        )
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, memory, ts_embeds, attn_mask=None):
        memory_input = memory
        memory = self.mut(memory, ts_embeds, attn_mask)
        memory = memory_input + self.norm1(self.drop_path1(memory))
        return memory


class TokenCrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv, attn_mask=None):
        B, N_q, C = q.shape
        _, N_kv, _ = kv.shape

        q_out = self.q_proj(q).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv_out = self.kv_proj(kv).reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k_out, v_out = kv_out.unbind(0)

        q_out, k_out = self.q_norm(q_out), self.k_norm(k_out)

        x = F.scaled_dot_product_attention(
            q_out, k_out, v_out, attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TokenCrossAttBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.attn = TokenCrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q, kv, attn_mask=None):
        q_input = q
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        x = self.attn(q_norm, kv_norm, attn_mask)
        return q_input + self.drop_path(x)


class DecoderBasicBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            prefix_num=10,
    ):
        super().__init__()

        self.prefix_num = prefix_num

        self.m_cross_x = TokenCrossAttBlock(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
            attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path, norm_layer=norm_layer
        )

        self.x_cross_m = TokenCrossAttBlock(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
            attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path, norm_layer=norm_layer
        )

        self.mtu_block = MTUBlock(
            dim=dim, drop_path=drop_path, norm_layer=norm_layer
        )

        self.feed_forward_prefix = nn.Sequential(
            norm_layer(dim),
            nn.Linear(dim, int(dim * mlp_ratio)),
            act_layer(),
            nn.Dropout(proj_drop),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

        self.feed_forward_instruct = nn.Sequential(
            norm_layer(dim),
            nn.Linear(dim, int(dim * mlp_ratio)),
            act_layer(),
            nn.Dropout(proj_drop),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, ts_embeds, attn_mask=None):

        memory_input = x[:, :self.prefix_num, :]
        x_input = x[:, self.prefix_num:, :]
        memory = self.m_cross_x(q=memory_input, kv=x_input, attn_mask=attn_mask)
        x = self.x_cross_m(q=x_input, kv=memory_input, attn_mask=attn_mask)
        x = self.feed_forward_instruct(x) + x
        memory = memory + self.mtu_block(memory, ts_embeds, attn_mask)
        memory = memory + self.feed_forward_prefix(memory)
        x = torch.cat([memory, x], dim=1)
        return x


class MFormer(nn.Module):
    def __init__(self, args):
        super(MFormer, self).__init__()
        self.layers = nn.ModuleList([
            DecoderBasicBlock(
                dim=args.m_d_model,
                num_heads=args.m_n_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_norm=False,
                proj_drop=args.m_dropout,
                attn_drop=args.m_dropout,
                drop_path=0.,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                prefix_num=args.prefix_num
            ) for _ in range(args.m_layers)
        ])
        self.norm = nn.LayerNorm(args.m_d_model)
        self.time_pos = SinusoidalPositionalEncoding(args.m_d_model)
        self.var_pos = LearnablePositionalEmbedding(args.m_d_model)
        self.instruc_pos = SinusoidalPositionalEncoding(args.m_d_model)
        self.cycle_pos = RotaryPositionalEncoding(args.m_d_model)
        self.prefix_num = args.prefix_num
        self.prefix_token = nn.Parameter(torch.randn(1, args.prefix_num, args.m_d_model))

    def forward(self, x, ts_embeds, stage=None, attn_mask=None):

        x = torch.cat([self.prefix_token.repeat(x.shape[0], 1, 1), x], dim=1)
        x = x + self.instruc_pos(x)

        # Stage 处理逻辑改进
        if torch.is_tensor(stage):
            stage_list = stage.tolist()
        else:
            stage_list = stage

        # 找出不同 stage 的索引
        cycle_index = [i for i, s in enumerate(stage_list) if s != 3 and s != 4]
        cross_cycle_index = [i for i, s in enumerate(stage_list) if s == 3 or s == 4]

        # 记录原始顺序以便恢复，确保与 x 对齐
        original_indices = cycle_index + cross_cycle_index
        reorder_map = {idx: i for i, idx in enumerate(original_indices)}
        reverse_indices = [reorder_map[i] for i in range(len(stage_list))]

        processed_memories = []

        if len(cycle_index) > 0:
            sub_ts_embeds = ts_embeds[cycle_index]
            b, l, v, d = sub_ts_embeds.shape
            sub_ts_embeds = sub_ts_embeds.view(b * l, v, d)
            sub_ts_embeds = sub_ts_embeds + self.time_pos(sub_ts_embeds)
            sub_ts_embeds = sub_ts_embeds.view(b, l, v, d)
            processed_memories.append((cycle_index, sub_ts_embeds))

        if len(cross_cycle_index) > 0:
            sub_ts_embeds = ts_embeds[cross_cycle_index]
            b, l, v, d = sub_ts_embeds.shape
            sub_ts_embeds = sub_ts_embeds.view(b * v, l, d)
            sub_ts_embeds = sub_ts_embeds + self.cycle_pos(sub_ts_embeds)
            sub_ts_embeds = sub_ts_embeds.view(b, l, v, d)
            processed_memories.append((cross_cycle_index, sub_ts_embeds))

        # 按照拼接后的顺序排列
        all_processed = torch.cat([m for _, m in processed_memories], dim=0)
        # 关键步骤：恢复原始 batch 顺序以匹配 x
        ts_embeds = all_processed[reverse_indices]

        # 再次处理变量维度
        b, l, v, d = ts_embeds.shape
        ts_embeds = ts_embeds.view(b * l, v, d)
        ts_embeds = ts_embeds + self.var_pos(ts_embeds)
        ts_embeds = ts_embeds.view(b, l, v, d)

        for i, layer in enumerate(self.layers):
            x = layer(x, ts_embeds, attn_mask)

        x = self.norm(x)
        return x[:, :self.prefix_num, :]


def count_parameters(model):
    """统计模型中可训练参数的总数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.m_d_model = 512
            self.m_n_heads = 8
            self.m_layers = 4
            self.m_dropout = 0.1
            self.prefix_num = 10

    args = Args()
    model = MFormer(args)

    # 打印可训练参数量
    total_trainable_params = count_parameters(model)
    print(f"Total Trainable Parameters: {total_trainable_params:,}")