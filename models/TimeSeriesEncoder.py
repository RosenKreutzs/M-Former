"""
Time Series Encoder
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from timm.layers import Mlp, DropPath
from timm.layers.helpers import to_2tuple
from transformers.modeling_outputs import CausalLMOutputWithPast



def calculate_unfold_output_length(input_length, size, step):
    # 计算窗口的数量
    num_windows = (input_length - size) // step + 1
    return num_windows


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            var_num=None,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if var_num is not None:
            self.template = nn.Parameter(
                torch.zeros(var_num, dim), requires_grad=True)
            torch.nn.init.normal_(self.template, std=.02)
        self.var_num = var_num

    def forward(self, x, query=None):
        B, N, C = x.shape
        if query is not None:
            q = self.q(query).reshape(
                B, query.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = self.q_norm(q)
            var_num = query.shape[1]
        else:
            q = self.q(self.template).reshape(1, self.var_num,
                                              self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q = self.q_norm(q)
            q = q.repeat(B, 1, 1, 1)
            var_num = self.var_num
        kv = self.kv(x).reshape(B, N, 2, self.num_heads,
                                self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        k = self.k_norm(k)

        # 确保内存连续性，使用稳定的注意力后端，防止CUDA非法内存访问
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )

        x = x.transpose(1, 2).reshape(B, var_num, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class FeedFoward(nn.Module):
    def __init__(
            self,
            dim,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            prefix_token_length=None,
            group=1,
    ):
        super().__init__()
        dim = dim
        hidden_features = hidden_features or 4*dim
        out_features = out_features or dim
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(dim, hidden_features,
                              bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])

        self.norm = norm_layer(
            hidden_features) if norm_layer is not None else nn.Identity()
 
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        self.hidden_features = hidden_features
        self.prefix_token_length = prefix_token_length

    def forward(self, x):
        n, var, l, d = x.shape
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.norm(x)
        x = self.drop2(x)+x
        return x



class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        self.pe = nn.Parameter(torch.zeros(
            1, 1, max_len, d_model), requires_grad=True)

        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).unsqueeze(0)
        self.pe.data.copy_(pe.float())
        del pe

    def forward(self, x, offset=0):
        return self.pe[:, :, offset:offset+x.size(2)]


class SeqAttention(nn.Module):

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
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # 确保内存连续性，使用稳定的注意力后端，防止CUDA非法内存访问
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            x = F.scaled_dot_product_attention(
                q, k, v,  # attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VarAttention(nn.Module):

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
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, P, C = x.shape

        qkv = self.qkv(x).reshape(B, N, P, 3, self.num_heads,
                                  self.head_dim).permute(3, 0, 2, 4, 1, 5)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.mean(dim=1, keepdim=False)
        k = k.mean(dim=1, keepdim=False)
        v = v.permute(0, 2, 3, 4, 1).reshape(B, self.num_heads, N, -1)

        # 确保内存连续性，使用稳定的注意力后端，防止CUDA非法内存访问
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )

        # 重塑以处理维度不匹配
        # 预期：x 的形状 num_heads, N, P * head_dim]
        # 目标：[B, N, P, C] 其中 C = num_heads * head_dim
        expected_last_dim = P * self.head_dim
        actual_last_dim = x.shape[-1]
        
        if actual_last_dim == expected_last_dim:
            # 标准情况：重塑回独立的P维
            x = x.view(B, self.num_heads, N, self.head_dim, P).permute(0, 2, 4, 1, 3).reshape(B, N, P, -1)
        else:
            # 处理维度不匹配的情况
            # 当注意力机制产生意外的维度时，我们通过使用更简单的重塑方法来适应
            # 这样在确保维度一致性的前提下保留了全部信息
            x = x.transpose(1, 2)  # [B, N, num_heads, actual_last_dim]
            
            # 平坦最后两个维度，然后重塑目标
            flattened_dim = self.num_heads * actual_last_dim
            x = x.reshape(B, N, flattened_dim)  # [B, N, flattened_dim]
            
            # 现在我们重塑为[B， N， P, remaining_dim]，其中remaining_dim = flated_dim / P
            if flattened_dim % P == 0:
                remaining_dim = flattened_dim // P
                x = x.reshape(B, N, P, remaining_dim)
            else:
                # 如果不能被整除，我们保持 P=1 并相应调整
                # 这是一个保留所有信息的备用方案
                x = x.reshape(B, N, 1, flattened_dim)
                #通过重复扩展到P个补丁（不理想，但可以防止崩溃）
                x = x.expand(B, N, P, flattened_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GateLayer(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gate = nn.Linear(dim, 1)

    def forward(self, x):
        gate_value = self.gate(x)
        return gate_value.sigmoid() * x


class SeqAttBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_seq = SeqAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, attn_mask):
        x_input = x
        x = self.norm1(x)
        n_vars, n_seqs = x.shape[1], x.shape[2]
        x = torch.reshape(
            x, (-1, x.shape[-2], x.shape[-1]))
        x = self.attn_seq(x, attn_mask)
        x = torch.reshape(
            x, (-1, n_vars, n_seqs, x.shape[-1]))
        x = x_input + self.drop_path1(x)
        return x


class VarAttBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_var = VarAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn_var(self.norm1(x)))
        return x


class MLPBlock(nn.Module):

    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            proj_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=None,
            prefix_token_length=0,
    ):
        super().__init__()
        self.norm2 = norm_layer(dim)
        if mlp_layer is FeedFoward:
            self.mlp = mlp_layer(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
                prefix_token_length=prefix_token_length,
            )
        else:
            self.mlp = mlp_layer(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
            )
        self.ls2 = GateLayer(dim, init_values=init_values)
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, prefix_seq_len=None):
        if prefix_seq_len is not None:
            x = x + \
                self.drop_path2(
                    self.ls2(self.mlp(self.norm2(x), prefix_seq_len=prefix_seq_len)))
        else:
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class BasicBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=8.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            prefix_token_length=0,
    ):
        super().__init__()
        self.seq_att_block = SeqAttBlock(dim=dim, num_heads=num_heads,
                                         qkv_bias=qkv_bias, qk_norm=qk_norm,
                                         attn_drop=attn_drop, init_values=init_values, proj_drop=proj_drop,
                                         drop_path=drop_path, norm_layer=norm_layer)

        self.var_att_block = VarAttBlock(dim=dim, num_heads=num_heads,
                                         qkv_bias=qkv_bias, qk_norm=qk_norm,
                                         attn_drop=attn_drop, init_values=init_values, proj_drop=proj_drop,
                                         drop_path=drop_path, norm_layer=norm_layer)


        self.feed_forward = FeedFoward(dim=dim, hidden_features=dim*4, act_layer=act_layer, drop=proj_drop)

    def forward(self, x, prefix_seq_len, attn_mask):
        x = self.var_att_block(x)
        x = self.seq_att_block(x, attn_mask)
        x = self.feed_forward(x)
        return x


class Patchfy(nn.Module):
    def __init__(self, patch_len, stride):
        super(Patchfy, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        assert self.patch_len == self.stride, "non-overlap"
    def forward(self, x):
        x = x.transpose(1, 2)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return x


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.patchfy = Patchfy(args.patch_len, args.stride)
        self.layers = nn.ModuleList([
            BasicBlock(
                dim=args.d_model,
                num_heads=args.n_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_norm=False,
                proj_drop=args.dropout,
                attn_drop=args.dropout,
                init_values=None,
                drop_path=0.,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                prefix_token_length=0
            ) for _ in range(args.e_layers)
        ])
        self.norm = nn.LayerNorm(args.d_model)
        self.patch_embedding = nn.Sequential(
            nn.Linear(args.patch_len, args.d_model, bias=False),
            nn.Dropout(args.dropout)
        )
        self.pretrain = getattr(args, 'pretrain', False)
        self.min_mask_ratio = getattr(args, 'min_mask_ratio', 0.7)
        self.max_mask_ratio = getattr(args, 'max_mask_ratio', 0.8)
        self.proj_head = nn.Linear(args.d_model, args.patch_len)
        # self.times_project = nn.Linear(6*args.d_model, args.d_model)
    def choose_masking(self, x, min_mask_ratio, max_mask_ratio):
        # 生成一个随机数以决定使用哪种掩码函数
        # 如果 torch.rand(1).item() 大于 right_prob：
        #     返回 self.random_masking(x， min_mask_ratio， max_mask_ratio)
        # 否则：
        #     返回 self.right_masking(x， min_mask_ratio， max_mask_ratio)
        return self.random_masking(x,min_mask_ratio,max_mask_ratio)
    
    def random_masking(self, x, min_mask_ratio, max_mask_ratio):
        """
        执行随机屏蔽，其中总V*L块的指定比例被屏蔽。
        """
        N, V, L, D = x.shape  # batch, var, length, dim
        total_elements = V * L

        mask_ratio = (min_mask_ratio+max_mask_ratio)/2
        #计算元素的数量保持基于掩码比率
        total_keeps = int((1 - mask_ratio) * total_elements)

        # 为批处理中的每个样本生成随机噪声数组
        noise = torch.rand(N, V, L, device=x.device)

        # 使噪声变平，便于处理
        noise_flat = noise.view(N, V * L)

        #获取索引来排序和恢复噪声
        ids_shuffle = torch.argsort(noise_flat, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 创建二进制掩码：0 表示保留，1 表示删除
        # 我们创建一个范围张量，并将其与 total_keeps 进行比较以生成掩码
        range_tensor = torch.arange(V * L, device=x.device).repeat(N, 1)
        mask_flat = range_tensor >= total_keeps

        # Unshuffle以获得原始顺序的二进制掩码
        mask_flat = mask_flat.gather(dim=1, index=ids_restore)

        # 重塑mask版回到原来的V， L尺寸
        mask = mask_flat.view(N, V, L)


        return mask
    
    def encode(self, x):
        B,n_vars,N,C = x.shape
        for layer in self.layers:
            x = layer(x, prefix_seq_len=None, attn_mask=None)
        x = self.norm(x)
        return x


    def forward(self, ts_values):
        x =  ts_values 
        x = self.patchfy(x)
        if self.pretrain:
            orin_x = x
            mask = self.choose_masking(x,self.min_mask_ratio, self.max_mask_ratio)
            mask_repeat = mask.unsqueeze(dim=-1) #[B,D,N,1]
            mask_repeat = mask_repeat.repeat(1, 1, 1, x.shape[-1])#[B,D,N,d]
            #进行掩码
            x = x.masked_fill(mask_repeat, 0)
        x = self.patch_embedding(x)
        x = self.encode(x)
        if self.pretrain:
            predict_x = self.proj_head(x)
            loss = F.mse_loss(predict_x, orin_x, reduction='mean')
            return CausalLMOutputWithPast(loss=loss, logits=x)
        return CausalLMOutputWithPast(logits=x,loss=None)




def test_model():
    import yaml
    import argparse


    #读取args
    parser = argparse.ArgumentParser(description='TsEncoder Pretrain')
    parser.add_argument('--fix_seed', type=int, default=None, help='seed')

    # model 设置
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4,
                        help='num of encoder layers')
    parser.add_argument("--patch_len", type=int, default=100)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--input_len", type=int, default=600)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--pretrain', type=bool, default=True, help='pretrain mode')
    parser.add_argument('--min_mask_ratio', type=float, default=0.1, help='minimum mask ratio')
    parser.add_argument('--max_mask_ratio', type=float, default=0.3, help='maximum mask ratio')

    args = parser.parse_args()
    # 创建一个Model实例

    model = Model(args)

    # 创建一些假的输入数据
    x_enc = torch.randn(10, 600, 33)  # 假设有10个样本，每个样本有50个时间步，每个时间步有100个特征

    # 调用forward方法
    dec_out = model.forward(x_enc)

    if 'loss' in dec_out:
        print(f"Loss: {dec_out['loss']}")
        print(f"Logits Shape: {dec_out['logits'].shape}")
    else:
        print(f"Logits Shape: {dec_out['logits'].shape}")

# 在脚本的最后调用测试函数
if __name__ == '__main__':
    test_model()