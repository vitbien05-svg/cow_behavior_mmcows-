import torch
import torch.nn as nn
import math


class SensorFCLayer(nn.Module):
    """
    FC Layer cho sensor features: 256 -> 128
    """

    def __init__(
        self, input_dim: int = 256, output_dim: int = 128, dropout: float = 0.2
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 256) hoặc (256,) nếu single sample
        Returns:
            (batch_size, 128) hoặc (128,)
        """
        x = self.norm(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


# ĐÃ SỬA LẠI CHO ĐÚNG BẢN GỐC
class ImageFCLayer(nn.Module):
    """
    FC Layer cho image features sau Global Pooling.
    Cấu trúc đúng theo paper:
        (256, 8, 8)
        -> GlobalAvgPool  -> (256,)
        -> GlobalMaxPool  -> (256,)
        -> concat         -> (512,)
        -> Normalize
        -> FC             -> (128,)
    """

    def __init__(
        self, input_dim: int = 512, output_dim: int = 128, dropout: float = 0.2
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),  # 512 -> 128, đúng theo paper
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(input_dim)  # normalize trước FC

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 256, 8, 8)  — feature tensor từ EfficientNet
        Returns:
            (batch_size, 128)
        """
        # Global pooling trên chiều spatial (H, W) — giữ nguyên 256 channels
        avg = x.mean(dim=[-2, -1])  # (batch_size, 256)
        mx = x.amax(dim=[-2, -1])  # (batch_size, 256)

        # Concat -> (batch_size, 512)
        out = torch.cat([avg, mx], dim=-1)

        # Normalize rồi FC
        out = self.norm(out)  # (batch_size, 512)
        out = self.fc(out)  # (batch_size, 128)
        return out


class FusionModule(nn.Module):
    """
    Fusion sensor (256,) + image (256, 8, 8) -> (256,) per timestep.
    Pipeline đúng theo paper:
        sensor: (256,) -> SensorFCLayer -> (128,)
        image:  (256, 8, 8) -> ImageFCLayer (global pool + FC) -> (128,)
        concat: (128,) + (128,) -> (256,)  -- đưa thẳng vào Transformer
    """

    def __init__(
        self,
        sensor_input_dim: int = 256,
        fusion_output_dim: int = 256,  # = 128 + 128, không đổi
        dropout: float = 0.2,
    ):
        super().__init__()
        self.sensor_fc = SensorFCLayer(sensor_input_dim, 128, dropout)
        self.image_fc = ImageFCLayer(input_dim=512, output_dim=128, dropout=dropout)
        # Không có fusion_fc — concat xong là xong

    def forward(
        self,
        sensor_features: torch.Tensor,  # (B, T, 256)
        image_features: torch.Tensor = None,  # (B, T, 256, 8, 8) hoặc None
    ) -> torch.Tensor:  # (B, T, 256)

        B, T, _ = sensor_features.shape

        # Flatten T vào batch để FC xử lý từng timestep song song
        s = sensor_features.reshape(B * T, -1)  # (B*T, 256)
        s = self.sensor_fc(s)  # (B*T, 128)

        if image_features is not None:
            i = image_features.reshape(
                B * T, *image_features.shape[2:]
            )  # (B*T, 256, 8, 8)
            i = self.image_fc(i)  # (B*T, 128)
        else:
            i = torch.zeros(
                B * T, 128, device=sensor_features.device, dtype=sensor_features.dtype
            )

        fused = torch.cat([i, s], dim=-1)  # (B*T, 256)
        return fused.reshape(B, T, 256)  # (B, T, 256)


# TRANSFORMER LAYERS
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding, inject thông tin vị trí vào sequence.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # dim chẵn
        pe[:, 1::2] = torch.cos(position * div_term)  # dim lẻ
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class FusionTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 1024,  # = 4 × d_model, gần convention Transformer gốc hơn
        dropout: float = 0.1,
        num_classes: int = 7,
        max_seq_len: int = 512,
    ):
        super().__init__()

        # ── Positional Encoding ──────────────────────────────────────────
        self.pos_encoder = PositionalEncoding(
            d_model, max_len=max_seq_len, dropout=dropout
        )

        # ── Transformer Encoder (2 lớp, 8 heads) ────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # 256
            nhead=nhead,  # 8
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # input shape: (batch, seq, feat)
            norm_first=False,  # Post-LN giống AIAYN gốc
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,  # 2
            norm=nn.LayerNorm(d_model),
        )

        # ── Classification Head ──────────────────────────────────────────
        # Dùng mean-pooling qua seq_len rồi qua FC → 5 classes
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # ── Khởi tạo trọng số ────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 256) — cat(image_feat, sensor_feat) đã align
            src_key_padding_mask: (batch, seq_len) bool mask, True = ignore
                                   Truyền vào nếu có padding trong batch.
        Returns:
            logits: (batch, num_classes)
        """
        # 1) Positional encoding
        x = self.pos_encoder(x)  # (B, T, 256)

        # 2) 2-layer Transformer Encoder
        enc_out = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )  # (B, T, 256)

        # 3) Temporal pooling: mean over seq_len (giống CLS mean-pooling)
        if src_key_padding_mask is not None:
            # Mask đi padding trước khi mean
            mask = (~src_key_padding_mask).unsqueeze(-1).float()  # (B, T, 1)
            pooled = (enc_out * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            pooled = enc_out.mean(dim=1)  # (B, 256)

        # 4) Classification
        logits = self.classifier(pooled)  # (B, 7)
        return logits


class CowBehaviorModel(nn.Module):
    """
    Full pipeline: Sensor + Image -> FusionModule -> FusionTransformer -> (B, 7)

    Kiến trúc:
        1. FusionModule   : (B,T,256) + (B,T,256,8,8) -> (B,T,256)
        2. FusionTransformer: (B,T,256) -> (B,7)
            - PositionalEncoding
            - 2-layer TransformerEncoder, 8 heads
            - mean pooling -> classifier
    """

    def __init__(
        self,
        num_classes: int = 7,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 1024,  # x4
        dropout: float = 0.2,
    ):
        super().__init__()

        self.fusion = FusionModule(
            sensor_input_dim=256,
            dropout=dropout,
        )
        self.transformer = FusionTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_classes=num_classes,
        )

    def forward(
        self,
        sensor: torch.Tensor,  # (B, T, 256)
        image: torch.Tensor = None,  # (B, T, 256, 8, 8) hoặc None
        src_key_padding_mask: torch.Tensor = None,  # (B, T) bool, True = padding
    ) -> torch.Tensor:  # (B, num_classes)

        fused = self.fusion(sensor, image)  # (B, T, 256)
        logits = self.transformer(fused, src_key_padding_mask)  # (B, 7)
        return logits


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def create_model(
    num_classes: int = 7,
    d_model: int = 256,
    device: str = "cpu",
) -> CowBehaviorModel:
    model = CowBehaviorModel(
        num_classes=num_classes,
        d_model=d_model,
        nhead=8,
        num_encoder_layers=2,
        dim_feedforward=1024,
        dropout=0.2,
    )
    return model.to(device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = create_model(num_classes=7, device=device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # Shape đúng theo paper: batch=32, seq_len=16
    sensor = torch.randn(32, 16, 256, device=device)
    image = torch.randn(32, 16, 256, 8, 8, device=device)

    logits = model(sensor, image)

    print(f"Sensor input : {sensor.shape}")  # (32, 16, 256)
    print(f"Image input  : {image.shape}")  # (32, 16, 256, 8, 8)
    print(f"Output logits: {logits.shape}")  # (32, 7)

    probs = torch.softmax(logits, dim=-1)
    print(f"Prob sum     : {probs[0].sum().item():.4f}")  # 1.0000
    print("Sanity check passed.")
