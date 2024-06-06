import torch
from torch import nn
import einops

class Class_Positions_Embeddings(nn.Module):
    def __init__(self, embed_dim, vocab_size, pad_value, chunk) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.chunk = chunk
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_value
        )
        self.positional_embedding = nn.Parameter(torch.rand(1, self.chunk, self.embed_dim))
        self.class_tokens = nn.Parameter(torch.rand(1, 1, self.embed_dim))

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_embedding.data
        batch, chunk, emd_dim = x.shape
        class_token = einops.repeat(self.class_tokens.data, "() chunk dim -> batch chunk dim", batch=batch)
        x = torch.cat((x, class_token), dim=1)
        return x

class MLP(nn.Module):
    def __init__(self, in_features:int, hidden_features=None, out_features=None, drop=0.0, act_layer = nn.GELU()):
        super().__init__()
        if out_features is None:
            out_features = in_features
        if hidden_features is None:
            hidden_features = in_features

        # Linear Layers
        self.lin1 = nn.Linear(
            in_features=in_features,
            out_features=hidden_features
        )
        self.lin2 = nn.Linear(
            in_features=hidden_features,
            out_features=out_features
        )

        # Activation(s)
        self.act = act_layer
        self.dropout = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.act(self.dropout(self.lin1(x)))
        x = self.act(self.lin2(x))

        return x

class Attention(nn.Module):
    def __init__(self, dim:int, num_heads:int, qkv_bias=False, attn_drop=0.0, out_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.soft = nn.Softmax(dim=-1) # Softmax по строкам матрицы внимания
        self.attn_drop = nn.Dropout(attn_drop)
        self.out = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(out_drop)

    def forward(self, x):
        # Attention
        qkv_after_linear = self.qkv(x)
        qkv_after_reshape = einops.rearrange(qkv_after_linear, "b c (v h w) -> v b h c w", v=3, h=self.num_heads)
        q = qkv_after_reshape[0]
        k = qkv_after_reshape[1]
        k = einops.rearrange(k, "b h c w -> b h w c") # Транспонирование
        v = qkv_after_reshape[2]

        atten = self.soft(torch.matmul(q, k) * self.scale)
        atten = self.attn_drop(atten)
        out = torch.matmul(atten, v)
        out = einops.rearrange(out, "b h c w -> b c (h w)", h=self.num_heads)

        # Out projection
        x = self.out(out)
        x = self.out_drop(x)

        return x

class Block(nn.Module):
    def __init__(self, dim:int, norm_type:str, num_heads:int, mlp_dim:int, qkv_bias=False, drop_rate=0.0):
        super().__init__()
        self.norm_type = norm_type

        # Normalization
        self.norm1 = nn.LayerNorm(
            normalized_shape=dim
        )
        self.norm2 = nn.LayerNorm(
            normalized_shape=dim
        )

        # Attention
        self.attention = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=drop_rate,
            out_drop=drop_rate
        )
        
        # MLP
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_dim
        )


    def forward(self, x):
        if self.norm_type == "prenorm":
            x_inner = self.norm1(x)
            # Attetnion
            x_inner = self.attention(x_inner)
            x = x_inner + x

            x_inner = self.norm2(x)
            # MLP
            x_inner = self.mlp(x_inner)
            x = x_inner + x
        
        if self.norm_type == "postnorm":
            x_inner = self.attention(x)
            x = x_inner + x
            x = self.norm1(x)
            x_inner = self.mlp(x)
            x = x_inner + x
            x =self.norm2(x)

        return x

class Transformer(nn.Module):
    def __init__(self, depth, dim, norm_type, num_heads, mlp_dim, qkv_bias=False, drop_rate=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, norm_type, num_heads, mlp_dim, qkv_bias, drop_rate) for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class SpecificBERT(nn.Module):
    def __init__(
            self, 
            vocab_size, embed_dim, pad_value, chunk_lenght,
            num_classes, depth, num_heads, mlp_dim,
            norm_type,
            qkv_bias=False, drop_rate=0.0
        ):
        super().__init__()
        # Позиционное кодирование и Эмбеддинги + Токен класса
        self.class_pos_emb = Class_Positions_Embeddings(
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            pad_value=pad_value,
            chunk=chunk_lenght
        )
        
        # Transformer Encoder
        self.transformer = Transformer(
            depth=depth,
            dim=embed_dim,
            norm_type=norm_type,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate
        )
        # Classifier
        self.head = MLP(
            in_features=embed_dim,
            out_features=num_classes,
            drop=drop_rate
        )

    def forward(self, x):
        x = self.class_pos_emb(x)
        x = self.transformer(x)
        x = self.head(x)
        return x