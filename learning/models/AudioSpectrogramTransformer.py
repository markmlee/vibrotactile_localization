import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=6, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

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

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class AST(nn.Module):
    def __init__(self, cfg, img_size=(50, 345), patch_size=(10, 10), in_chans=6, num_outputs=3, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.num_outputs = num_outputs
        self.embed_dim = embed_dim
        self.num_tokens = 1

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer
            )
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Regression head
        self.head = nn.Linear(embed_dim, num_outputs)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.pos_embed, std=.02)
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)

        # Use the cls token for prediction
        x = x[:, 0]

        x = self.head(x)
        return x



class AST_multimodal(nn.Module):
    def __init__(self, cfg, img_size=(50, 345), patch_size=(10, 10), in_chans=6, num_outputs=3,
                 depth=12, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        
        # Configuration validation
        self.qt_seq_len = 100  # Sequence length for trajectory
        self.qt_dim = 7  # 7 trajectories per time step
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        
        # Different embedding dimensions for audio vs other modalities
        self.audio_embed_dim = 768    # Larger embedding for audio
        self.other_embed_dim = 384    # Smaller for qt and tdoa
        
        # Different number of attention heads
        self.audio_num_heads = 12     # More heads for audio processing
        self.other_num_heads = 6      # Fewer heads for other modalities
        
        # Different dropout rates
        self.audio_drop_rate = 0.1    # Less dropout for audio
        self.other_drop_rate = 0.3    # More dropout for others

        self.num_outputs = num_outputs
        self.num_tokens = 1

        

        # Validate input dimensions
        assert len(img_size) == 2, f"Expected img_size to have 2 dimensions, got {len(img_size)}"
        assert len(patch_size) == 2, f"Expected patch_size to have 2 dimensions, got {len(patch_size)}"

        # 1. Audio Spectrogram Path
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=self.audio_embed_dim
        )
        num_patches = self.patch_embed.n_patches

        # Calculate total number of tokens correctly
        total_tokens = (
            1 +             # CLS token
            1 +             # audio token
            num_patches +   # audio patches
            1 +             # qt token
            1 +             # qt embedding
            1 +             # tdoa token
            1              # tdoa embedding
        )
        
        # Position embeddings with correct size
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, self.audio_embed_dim))

        print(f"Token count verification:")
        print(f"CLS token: 1")
        print(f"Audio token: 1")
        print(f"Audio patches: {num_patches}")
        print(f"QT token + embedding: 2")
        print(f"TDOA token + embedding: 2")
        print(f"Total tokens: {total_tokens}")

        # 2. Robot Joint Trajectory Path with sequence processing
        self.qt_embedding = nn.Sequential(
            # First embed each time step's joint values
            nn.Linear(self.qt_dim, self.other_embed_dim),
            nn.LayerNorm(self.other_embed_dim),
            nn.Dropout(self.other_drop_rate),
        )
        
        # Add sequence processing layer
        self.qt_sequence_processor = nn.Sequential(
            # Process the sequence using a small transformer or temporal convolution
            nn.Linear(self.qt_seq_len * self.other_embed_dim, self.audio_embed_dim),
            nn.LayerNorm(self.audio_embed_dim),
            nn.Dropout(self.other_drop_rate)
        )

        # 3. TDOA Path
        self.tdoa_dim = 15  # 15 pairs from 6 microphones
        self.tdoa_embedding = nn.Sequential(
            nn.Linear(self.tdoa_dim, self.other_embed_dim),
            nn.LayerNorm(self.other_embed_dim),
            nn.Dropout(self.other_drop_rate),
            nn.Linear(self.other_embed_dim, self.audio_embed_dim)
        )

        # Add normalization layers for qt and tdoa
        self.qt_norm = nn.LayerNorm(self.qt_dim)    # Normalize each joint value
        self.tdoa_norm = nn.LayerNorm(self.tdoa_dim) # Normalize TDOA values


        # Learnable tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.audio_embed_dim))
        self.audio_token = nn.Parameter(torch.zeros(1, 1, self.audio_embed_dim))
        self.qt_token = nn.Parameter(torch.zeros(1, 1, self.audio_embed_dim))
        self.tdoa_token = nn.Parameter(torch.zeros(1, 1, self.audio_embed_dim))

        # Dropout layer
        self.pos_drop = nn.Dropout(p=self.audio_drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=self.audio_embed_dim,
                num_heads=self.audio_num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop=self.audio_drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer
            )
            for _ in range(self.depth)])
        
        self.norm = norm_layer(self.audio_embed_dim)

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(self.audio_embed_dim, self.audio_embed_dim // 2),
            nn.GELU(),
            nn.Dropout(self.audio_drop_rate),
            nn.Linear(self.audio_embed_dim // 2, num_outputs)
        )

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize all learnable parameters
        """
        # Initialize position embeddings
        torch.nn.init.normal_(self.pos_embed, std=.02)
        
        # Initialize special tokens
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.audio_token, std=.02)
        torch.nn.init.normal_(self.qt_token, std=.02)
        torch.nn.init.normal_(self.tdoa_token, std=.02)
        
        # Apply _init_weights to all other parameters
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights for different layer types
        """
        if isinstance(m, nn.Linear):
            # Initialize linear layers
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # Initialize layer normalization
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # Initialize convolutional layers (if any)
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, qt, tdoa):
        # Input validation
        assert x.dim() == 4, f"Expected 4D tensor for audio input, got {x.dim()}D"
        assert qt.dim() == 3, f"Expected 3D tensor for qt input [batch, 100, 7], got {qt.dim()}D"
        assert qt.shape[1:] == (self.qt_seq_len, self.qt_dim), f"Expected qt shape [..., 100, 7], got [..., {qt.shape[1]}, {qt.shape[2]}]"
        assert tdoa.dim() == 2, f"Expected 2D tensor for tdoa input, got {tdoa.dim()}D"
        
        B = x.shape[0]

        # 1. Process audio
        x = self.patch_embed(x)

        # 2. Process qt trajectory sequence with normalization
        # Normalize each time step's joint values
        qt_reshaped = qt.view(-1, self.qt_dim)  # Reshape to [B*100, 7]
        qt_normalized = self.qt_norm(qt_reshaped)  # Apply normalization
        qt_normalized = qt_normalized.view(B, self.qt_seq_len, self.qt_dim)  # Reshape back

        # Continue with qt processing
        qt = self.qt_embedding(qt_normalized)
        qt_flat = qt.reshape(B, -1)
        qt_embedded = self.qt_sequence_processor(qt_flat)
        qt_embedded = qt_embedded.unsqueeze(1)

        # 3. Process TDOA
        tdoa_normalized = self.tdoa_norm(tdoa)
        tdoa_embedded = self.tdoa_embedding(tdoa_normalized).unsqueeze(1)

        # Expand tokens
        cls_token = self.cls_token.expand(B, -1, -1)
        audio_token = self.audio_token.expand(B, -1, -1)
        qt_token = self.qt_token.expand(B, -1, -1)
        tdoa_token = self.tdoa_token.expand(B, -1, -1)

        # Debug print shapes before concatenation
        # print(f"Shape debug:")
        # print(f"cls_token shape: {cls_token.shape}")
        # print(f"audio_token shape: {audio_token.shape}")
        # print(f"x shape: {x.shape}")
        # print(f"qt_token shape: {qt_token.shape}")
        # print(f"qt_embedded shape: {qt_embedded.shape}")
        # print(f"tdoa_token shape: {tdoa_token.shape}")
        # print(f"tdoa_embedded shape: {tdoa_embedded.shape}")

        # Concatenate all tokens and embeddings
        x = torch.cat((
            cls_token, audio_token, x,
            qt_token, qt_embedded,
            tdoa_token, tdoa_embedded
        ), dim=1)

        # print(f"Concatenated x shape: {x.shape}")
        # print(f"Position embedding shape: {self.pos_embed.shape}")

        # Add position embeddings and apply transformer
        x = self.pos_drop(x + self.pos_embed)
        
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Use cls token for prediction
        x = self.head(x[:, 0])
        
        return x
    

class AST_multimodal_qt(nn.Module):
    def __init__(self, cfg, img_size=(50, 345), patch_size=(10, 10), in_chans=6, num_outputs=3,
                 depth=12, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        
        # Configuration validation
        self.qt_seq_len = 100  # Sequence length for trajectory
        self.qt_dim = 7        # 7 trajectories per time step
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        
        # Different embedding dimensions for audio vs other modalities
        self.audio_embed_dim = 768    # Larger embedding for audio
        self.other_embed_dim = 384    # Smaller for qt
        
        # Different number of attention heads
        self.audio_num_heads = 12     # More heads for audio processing
        self.other_num_heads = 6      # Fewer heads for other modalities
        
        # Different dropout rates
        self.audio_drop_rate = 0.1    # Less dropout for audio
        self.other_drop_rate = 0.3    # More dropout for others

        self.num_outputs = num_outputs
        self.num_tokens = 1

        # Validate input dimensions
        assert len(img_size) == 2, f"Expected img_size to have 2 dimensions, got {len(img_size)}"
        assert len(patch_size) == 2, f"Expected patch_size to have 2 dimensions, got {len(patch_size)}"

        # 1. Audio Spectrogram Path
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=self.audio_embed_dim
        )
        num_patches = self.patch_embed.n_patches

        # Calculate total number of tokens correctly
        total_tokens = (
            1 +             # CLS token
            1 +             # audio token
            num_patches +   # audio patches
            1 +             # qt token
            1              # qt embedding
        )
        
        # Position embeddings with correct size
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, self.audio_embed_dim))

        print(f"Token count verification:")
        print(f"CLS token: 1")
        print(f"Audio token: 1")
        print(f"Audio patches: {num_patches}")
        print(f"QT token + embedding: 2")
        print(f"Total tokens: {total_tokens}")

        # 2. Robot Joint Trajectory Path with sequence processing
        self.qt_embedding = nn.Sequential(
            # First embed each time step's joint values
            nn.Linear(self.qt_dim, self.other_embed_dim),
            nn.LayerNorm(self.other_embed_dim),
            nn.Dropout(self.other_drop_rate),
        )
        
        # Add sequence processing layer
        self.qt_sequence_processor = nn.Sequential(
            # Process the sequence using a small transformer or temporal convolution
            nn.Linear(self.qt_seq_len * self.other_embed_dim, self.audio_embed_dim),
            nn.LayerNorm(self.audio_embed_dim),
            nn.Dropout(self.other_drop_rate)
        )

        # Add normalization layer for qt
        self.qt_norm = nn.LayerNorm(self.qt_dim)    # Normalize each joint value

        # Learnable tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.audio_embed_dim))
        self.audio_token = nn.Parameter(torch.zeros(1, 1, self.audio_embed_dim))
        self.qt_token = nn.Parameter(torch.zeros(1, 1, self.audio_embed_dim))

        # Dropout layer
        self.pos_drop = nn.Dropout(p=self.audio_drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=self.audio_embed_dim,
                num_heads=self.audio_num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop=self.audio_drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer
            )
            for _ in range(self.depth)])
        
        self.norm = norm_layer(self.audio_embed_dim)

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(self.audio_embed_dim, self.audio_embed_dim // 2),
            nn.GELU(),
            nn.Dropout(self.audio_drop_rate),
            nn.Linear(self.audio_embed_dim // 2, num_outputs)
        )

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize all learnable parameters"""
        torch.nn.init.normal_(self.pos_embed, std=.02)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.audio_token, std=.02)
        torch.nn.init.normal_(self.qt_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights for different layer types"""
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, qt):
        # Input validation
        assert x.dim() == 4, f"Expected 4D tensor for audio input, got {x.dim()}D"
        assert qt.dim() == 3, f"Expected 3D tensor for qt input [batch, 100, 7], got {qt.dim()}D"
        assert qt.shape[1:] == (self.qt_seq_len, self.qt_dim), f"Expected qt shape [..., 100, 7], got [..., {qt.shape[1]}, {qt.shape[2]}]"
        
        B = x.shape[0]

        # 1. Process audio (no normalization)
        x = self.patch_embed(x)

        # 2. Process qt trajectory sequence with normalization
        qt_reshaped = qt.view(-1, self.qt_dim)  # Reshape to [B*100, 7]
        qt_normalized = self.qt_norm(qt_reshaped)  # Apply normalization
        qt_normalized = qt_normalized.view(B, self.qt_seq_len, self.qt_dim)  # Reshape back

        # Continue with qt processing
        qt = self.qt_embedding(qt_normalized)
        qt_flat = qt.reshape(B, -1)
        qt_embedded = self.qt_sequence_processor(qt_flat)
        qt_embedded = qt_embedded.unsqueeze(1)

        # Expand tokens
        cls_token = self.cls_token.expand(B, -1, -1)
        audio_token = self.audio_token.expand(B, -1, -1)
        qt_token = self.qt_token.expand(B, -1, -1)

        # Concatenate all tokens and embeddings
        x = torch.cat((
            cls_token, audio_token, x,
            qt_token, qt_embedded
        ), dim=1)

        # Add position embeddings and apply transformer
        x = self.pos_drop(x + self.pos_embed)
        
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Use cls token for prediction
        x = self.head(x[:, 0])
        
        return x