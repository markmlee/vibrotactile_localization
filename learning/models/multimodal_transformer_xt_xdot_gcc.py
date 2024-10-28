import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, hidden_dim=dim * mlp_ratio, dropout=dropout)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out += self.shortcut(identity)
        out = F.relu(out)
        return out

class ResNet50_Feature(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.conv1 = nn.Conv2d(cfg.num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.apply(self._init_weights)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(Bottleneck(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_channels * 4, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self, m):  # m is passed by apply()
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class StateEncoder(nn.Module):
    def __init__(self, embedding_dim=128):  # Reduced embedding dimension
        super().__init__()
        
        self.input_norm = nn.BatchNorm1d(10)  # 7 pos + 3 vel = 10 channels
        
        # Reduced capacity architecture
        self.conv1d = nn.Sequential(
            nn.Conv1d(10, 16, kernel_size=5, stride=2),  # Changed input channels to 10
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),  # Higher dropout
            
            nn.Conv1d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(32, embedding_dim, kernel_size=5, stride=2),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x = x.transpose(1, 2)  # [batch_size, 10, timesteps]
        x = self.input_norm(x)
        x = self.conv1d(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        return x
    
class GCCPHATEncoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        
        # Input shape: [batch_size, 15, 883]
        self.input_norm = nn.LayerNorm(15 * 883)  # Normalize flattened input
        
        # MLP to preserve exact time delay values
        self.mlp = nn.Sequential(
            nn.Linear(15 * 883, 1024),
            nn.LayerNorm(1024),  # Normalize intermediate features
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, embedding_dim)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        # Flatten preserving batch dimension: [B, 15, 883] -> [B, 15*883]
        x = x.view(batch_size, -1)
        x = self.input_norm(x)
        x = self.mlp(x)
        return x
    
class MultiModalTransformer_xt_xdot_t_gccphat(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Architecture parameters
        self.embedding_dim = 256
        self.audio_embed_dim = self.embedding_dim  # Full dimension for audio
        self.gcc_embed_dim = self.embedding_dim    # Full dimension for GCC-PHAT
        self.state_embed_dim = self.embedding_dim // 2  # Half dimension for state
        self.num_heads = 8
        self.num_layers = 6
        self.mlp_ratio = 4
        
        # Different dropout rates
        self.audio_dropout = 0.1  # Lower dropout for audio
        self.gcc_dropout = 0.1    # Lower dropout for GCC-PHAT
        self.state_dropout = 0.3  # Higher dropout for state
        
        # Audio encoder with larger capacity
        self.audio_encoder = ResNet50_Feature(cfg)
        self.audio_projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(self.audio_dropout),
            nn.Linear(1024, self.audio_embed_dim)
        )
        
        # GCC-PHAT encoder using MLP
        self.gcc_encoder = GCCPHATEncoder(embedding_dim=self.gcc_embed_dim)
        
        # State encoder with smaller capacity
        self.state_encoder = StateEncoder(embedding_dim=self.state_embed_dim)
        
        # Modal encodings
        self.modal_enc_audio = nn.Parameter(torch.randn(1, 1, self.audio_embed_dim))
        self.modal_enc_gcc = nn.Parameter(torch.randn(1, 1, self.gcc_embed_dim))
        self.modal_enc_state = nn.Parameter(torch.randn(1, 1, self.state_embed_dim))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        
        # Layer Normalization
        self.input_norm = nn.LayerNorm(self.embedding_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.embedding_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                dropout=self.audio_dropout
            )
            for _ in range(self.num_layers)
        ])
        
        # Output head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(self.audio_dropout),
            nn.Linear(self.embedding_dim * 2, 3)  # height, x, y
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, audio_input, state_input, gcc_input):
        """
        Args:
            audio_input: Tensor of shape [batch_size, channels, freq, time]
            state_input: Tensor of shape [batch_size, timesteps, 10] (7 pos + 3 vel)
            gcc_input: Tensor of shape [batch_size, 15, 883] (15 mic pairs)
        """
        batch_size = audio_input.shape[0]
        
        # Verify input shapes
        assert gcc_input.shape[1] == 15, f"Expected 15 mic pairs, got {gcc_input.shape[1]}"
        assert gcc_input.shape[2] == 883, f"Expected 883 time points, got {gcc_input.shape[2]}"
        assert state_input.shape[2] == 10, f"Expected 10 state dimensions, got {state_input.shape[2]}"
        
        # Encode audio
        audio_features = self.audio_encoder(audio_input)
        audio_embed = self.audio_projection(audio_features)
        audio_embed = audio_embed.unsqueeze(1)  # [B, 1, 256]
        
        # Encode GCC-PHAT (preserving exact time delays)
        gcc_embed = self.gcc_encoder(gcc_input)
        gcc_embed = gcc_embed.unsqueeze(1)  # [B, 1, 256]
        
        # Encode state trajectories
        state_embed = self.state_encoder(state_input)
        state_embed = state_embed.unsqueeze(1)  # [B, 1, 128]
        
        # Add modal encodings
        audio_embed = audio_embed + self.modal_enc_audio
        gcc_embed = gcc_embed + self.modal_enc_gcc
        state_embed = state_embed + self.modal_enc_state
        
        # Project state embeddings to match audio dimension
        state_embed = F.pad(state_embed, (0, self.audio_embed_dim - self.state_embed_dim))
        
        # Concatenate CLS token and embeddings
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        x = torch.cat([cls_tokens, audio_embed, gcc_embed, state_embed], dim=1)
        
        # Input normalization
        x = self.input_norm(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Get CLS token output and predict
        x = x[:, 0]
        output = self.mlp_head(x)
        
        return output