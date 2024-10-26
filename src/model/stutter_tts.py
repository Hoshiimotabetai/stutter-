import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from .encoder import Encoder, GlobalAudioReferenceEncoder
from .decoder import Decoder
import math

class StutterTTS(nn.Module):
    """Stutter-TTS: 吃音パターンを制御可能なTTSモデル"""
    def __init__(self, config: Dict):
        super().__init__()
        
        # 設定の取得
        model_config = config["model"]
        d_model = model_config["d_model"]
        n_heads = model_config["n_heads"]
        n_encoder_layers = model_config["n_encoder_layers"]
        n_decoder_layers = model_config["n_decoder_layers"]
        d_ff = model_config["d_ff"]
        dropout = model_config["dropout"]
        n_mel_channels = config["data"]["n_mels"]
        
        # エンコーダー
        self.encoder = Encoder(
            num_phonemes=model_config["n_phonemes"],
            d_model=d_model,
            n_head=n_heads,
            d_ff=d_ff,
            num_layers=n_encoder_layers,
            dropout=dropout
        )
        
        # グローバルオーディオリファレンスエンコーダー
        self.reference_encoder = GlobalAudioReferenceEncoder(
            mel_channels=n_mel_channels,
            d_model=d_model
        )
        
        # デコーダー
        self.decoder = Decoder(
            mel_channels=n_mel_channels,
            d_model=d_model,
            n_head=n_heads,
            d_ff=d_ff,
            num_layers=n_decoder_layers,
            prenet_hidden_dim=d_model // 2,
            prenet_dropout=0.6,
            decoder_dropout=dropout
        )
        
        # 初期化
        self._reset_parameters()

    def _reset_parameters(self):
        """モデルパラメータの初期化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_masks(self,
                     src_seq: torch.Tensor,
                     tgt_seq: Optional[torch.Tensor] = None,
                     src_pad_idx: int = 0,
                     tgt_pad_idx: int = 0) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """マスクの作成
        Args:
            src_seq: ソース系列（音素ID）
            tgt_seq: ターゲット系列（メルスペクトログラム）
            src_pad_idx: ソースのパディングインデックス
            tgt_pad_idx: ターゲットのパディングインデックス
        """
        # エンコーダーパディングマスク
        src_mask = (src_seq != src_pad_idx).unsqueeze(1).unsqueeze(2).to(torch.float32)
        
        # デコーダーマスク（学習時のみ使用）
        if tgt_seq is not None:
            tgt_len=tgt_seq.size(2)
            tgt_pad_mask = torch.ones((tgt_seq.size(0),1, tgt_len, tgt_len),
                                      device=tgt_seq.device,
                                      dtype=torch.float32)
        #    seq_length = tgt_seq.size(1)
            subsequent_mask = (torch.triu(
                torch.ones((tgt_len, tgt_len), device=tgt_seq.device),
                diagonal=1
            ) == 1)
            tgt_mask = tgt_pad_mask * (1.0 - subsequent_mask.float())
        else:
            tgt_mask = None
            
        return src_mask, tgt_mask

    def forward(self,
                phoneme_ids: torch.Tensor,
                mel_target: Optional[torch.Tensor] = None,
                reference_mel: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            phoneme_ids: 音素ID系列 [batch_size, src_len]
            mel_target: 目標メルスペクトログラム [batch_size, mel_channels, tgt_len]（学習時のみ）
            reference_mel: リファレンス音声のメルスペクトログラム [batch_size, mel_channels, ref_len]
        Returns:
            mel_output: 生成されたメルスペクトログラム
            mel_postnet: Postnet適用後のメルスペクトログラム
            stop_tokens: 停止トークン予測
        """
        # マスクの作成
        src_mask, tgt_mask = self._create_masks(
            phoneme_ids,
            mel_target#.transpose(1, 2) if mel_target is not None else None
        )
        
        # エンコーダー処理
        encoder_output, phoneme_embeddings = self.encoder(phoneme_ids, src_mask)
        
        # リファレンスエンコーディング
        if reference_mel is not None:
            reference_embedding = self.reference_encoder(reference_mel)
            # リファレンス埋め込みをエンコーダー出力に追加
            encoder_output = encoder_output + reference_embedding.unsqueeze(1)
        
        # デコーダー処理
        if mel_target is not None:
            mel_target = mel_target.transpose(1, 2)

        mel_output, mel_postnet, stop_tokens = self.decoder(
            encoder_output=encoder_output,
            mel_target=mel_target.transpose(1, 2) if mel_target is not None else None,
            encoder_padding_mask=src_mask,
            self_attention_mask=tgt_mask
        )
        """
        # 出力の転置（batch, mel_channels, time）の形式に
        if mel_output.size(2) != mel_target.size(2) and mel_target is not None:
            # サイズの不一致がある場合は切り詰めまたはパディング
            target_length = mel_target.size(2)
            mel_output = F.pad(
                mel_output.transpose(1, 2),
                (0, max(0, target_length - mel_output.size(1)))
            )[:, :target_length]
            mel_postnet = F.pad(
                mel_postnet.transpose(1, 2),
                (0, max(0, target_length - mel_postnet.size(1)))
            )[:, :target_length]
        else:
            mel_output = mel_output.transpose(1, 2)
            mel_postnet = mel_postnet.transpose(1, 2)
        """
        # 出力の転置（batch, mel_channels, time）の形式に
        if mel_target is not None:
            #target_length = mel_target.size(2)
            # サイズの調整
            mel_output = mel_output.transpose(1, 2)
            mel_postnet = mel_postnet.transpose(1, 2)
            
            # 必要に応じて切り詰めまたはパディング
            if mel_output.size(2) != tgt_len:
                mel_output = F.pad(
                    mel_output,
                    (0, max(0, tgt=len - mel_output.size(2)))
                )[:, :, :tgt_len]
                
                mel_postnet = F.pad(
                    mel_postnet,
                    (0, max(0, tgt_len - mel_postnet.size(2)))
                )[:, :, :tgt_len]
        
        return mel_output, mel_postnet, stop_tokens

class StutterTTSLoss(nn.Module):
    """Stutter-TTSの損失関数"""
    def __init__(self):
        super().__init__()

    def forward(self,
                mel_output: torch.Tensor,
                mel_postnet: torch.Tensor,
                stop_tokens: torch.Tensor,
                mel_target: torch.Tensor,
                stop_target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            mel_output: デコーダー出力のメルスペクトログラム
            mel_postnet: Postnet適用後のメルスペクトログラム
            stop_tokens: 停止トークン予測
            mel_target: 目標メルスペクトログラム
            stop_target: 目標停止トークン
        Returns:
            total_loss: 総損失
            loss_dict: 各損失コンポーネントの辞書
        """
        # メルスペクトログラムの損失（L1損失）
        mel_loss = F.l1_loss(mel_output, mel_target)
        postnet_mel_loss = F.l1_loss(mel_postnet, mel_target)
        
        # 停止トークンの損失（バイナリクロスエントロピー）
        stop_loss = F.binary_cross_entropy(
            stop_tokens.squeeze(-1),
            stop_target.float()
        )
        
        # 総損失
        total_loss = mel_loss + postnet_mel_loss + stop_loss
        
        # 損失の詳細を辞書で返す
        loss_dict = {
            'mel_loss': mel_loss.item(),
            'postnet_mel_loss': postnet_mel_loss.item(),
            'stop_loss': stop_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict

def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """後続マスクの生成（推論時用）"""
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
    mask = mask.float().masked_fill(mask == 1, float('-inf'))
    return mask

@torch.no_grad()
def inference(model: StutterTTS,
              phoneme_ids: torch.Tensor,
              reference_mel: Optional[torch.Tensor] = None,
              max_length: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """推論用の関数
    Args:
        model: StutterTTSモデル
        phoneme_ids: 音素ID系列
        reference_mel: リファレンス音声（オプション）
        max_length: 最大生成長
    Returns:
        mel_postnet: 生成されたメルスペクトログラム
        stop_tokens: 停止トークン
    """
    model.eval()
    
    device = next(model.parameters()).device
    phoneme_ids = phoneme_ids.to(device)
    if reference_mel is not None:
        reference_mel = reference_mel.to(device)
    
    # エンコーダーマスクの作成
    src_mask = (phoneme_ids != 0).unsqueeze(1).unsqueeze(2)
    
    # 推論実行
    mel_output, mel_postnet, stop_tokens = model(
        phoneme_ids=phoneme_ids,
        reference_mel=reference_mel
    )
    
    return mel_postnet, stop_tokens

def count_parameters(model: nn.Module) -> int:
    """モデルのパラメータ数を計算"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# モデルのテスト用コード
if __name__ == "__main__":
    import yaml
    from pathlib import Path
    
    # 設定ファイルの読み込み
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # モデルの作成
    model = StutterTTS(config)
    
    # パラメータ数の表示
    print(f"Number of parameters: {count_parameters(model):,}")
    
    # 入力テンソルの作成（テスト用）
    batch_size = 2
    src_len = 50
    tgt_len = 100
    n_mel = config["data"]["n_mels"]
    
    phoneme_ids = torch.randint(0, 100, (batch_size, src_len))
    mel_target = torch.randn(batch_size, n_mel, tgt_len)
    reference_mel = torch.randn(batch_size, n_mel, 200)
    
    # モデルの実行
    mel_output, mel_postnet, stop_tokens = model(
        phoneme_ids=phoneme_ids,
        mel_target=mel_target,
        reference_mel=reference_mel
    )
    
    # 出力サイズの表示
    print(f"Mel output shape: {mel_output.shape}")
    print(f"Mel postnet shape: {mel_postnet.shape}")
    print(f"Stop tokens shape: {stop_tokens.shape}")