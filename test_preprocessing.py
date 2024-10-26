# test_preprocessing.py
import yaml
from pathlib import Path
from src.data import StutterDataModule
import torch
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_single_samples(data_module):
    #-----------------------------#
    """単一サンプルの処理をテスト"""
    print("\n=== Testing Single Sample Processing ===")
    
    # テキスト処理のテスト
    # LibriSpeechからサンプルテキストを取得
    librispeech_path = Path(data_module.config["data"]["librispeech_path"])
    trans_file = next(librispeech_path.rglob("*.trans.txt"))
    with open(trans_file, "r") as f:
        sample_line = f.readline().strip()
    text = sample_line.split(" ", 1)[1]  # ID部分を除去
    
    text_data = data_module.process_single_text(text)
    print("\nText Processing Test:")
    print(f"Original text: {text}")
    print(f"Number of phonemes: {len(text_data['phonemes'])}")
    print(f"First few phonemes: {text_data['phonemes'][:10]}")
    print(f"Phoneme IDs shape: {len(text_data['phoneme_ids'])}")

    # 音声処理のテスト
    audio_file = next(librispeech_path.rglob("*.flac"))
    print(f"\nProcessing audio file: {audio_file}")
    
    try:
        audio_data = data_module.process_single_audio(str(audio_file))
        print("\nAudio Processing Results:")
        print(f"Waveform shape: {audio_data['waveform'].shape}")
        print(f"Mel spectrogram shape: {audio_data['mel_spectrogram'].shape}")
        print(f"Duration: {audio_data['duration']:.2f} seconds")
        
        # 音声データの可視化
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(audio_data["waveform"].squeeze().numpy())
        plt.title("Waveform")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        
        plt.subplot(1, 2, 2)
        librosa.display.specshow(
            audio_data["mel_spectrogram"].squeeze().numpy(),
            y_axis='mel',
            x_axis='time',
            sr=data_module.config["data"]["sample_rate"],
            hop_length=data_module.config["data"]["hop_length"]
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title("Mel Spectrogram")
        plt.tight_layout()
        
        # 出力ディレクトリの作成
        output_dir = Path("test_outputs")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "preprocessing_test_single.png")
        plt.close()
        
        print("\nVisualization saved to test_outputs/preprocessing_test_single.png")
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        raise

    return audio_data, text_data

    #-----------------------------#
    """単一サンプルの処理をテスト"""
    """
    # テキスト処理のテスト
    text = "This is a test sentence"
    text_data = data_module.process_single_text(text)
    print("\nText Processing Test:")
    print(f"Original text: {text}")
    print(f"Phoneme IDs: {text_data['phoneme_ids']}")
    print(f"Phonemes: {text_data['phonemes']}")

    # 吃音イベント付きのテスト
    stutter_events = [{"type": "repetition", "position": 1}]
    text_data_with_stutter = data_module.process_single_text(text, stutter_events)
    print("\nText Processing with Stutter Test:")
    print(f"Stutter events: {stutter_events}")
    print(f"Phoneme IDs: {text_data_with_stutter['phoneme_ids']}")

    # 音声処理のテスト
    # LibriSpeechの最初のファイルを使用
    librispeech_path = Path(data_module.config["data"]["librispeech_path"])
    audio_file = next(librispeech_path.rglob("*.flac"))
    audio_data = data_module.process_single_audio(str(audio_file))
    
    # 音声データの可視化
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(audio_data["waveform"].squeeze().numpy())
    plt.title("Waveform")
    
    plt.subplot(1, 2, 2)
    librosa.display.specshow(
        audio_data["mel_spectrogram"].squeeze().numpy(),
        y_axis='mel', x_axis='time'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.savefig("preprocessing_test_single.png")
    plt.close()

    return audio_data, text_data
"""
def test_batch_processing(data_module):
    """バッチ処理のテスト"""
    #-----------------#
    """バッチ処理のテスト"""
    print("\n=== Testing Batch Processing ===")
    
    try:
        train_loader = data_module.train_loader
        batch = next(iter(train_loader))
        
        print("\nBatch Information:")
        print(f"Batch keys: {batch.keys()}")
        print(f"Mel spectrograms shape: {batch['mel_spectrogram'].shape}")
        print(f"Phoneme IDs shape: {batch['phoneme_ids'].shape}")
        print(f"Number of samples in batch: {len(batch['texts'])}")
        
        # バッチデータの可視化
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(batch['mel_spectrogram'][0].numpy(), aspect='auto')
        plt.title("First sample in batch (Mel Spectrogram)")
        plt.xlabel("Time")
        plt.ylabel("Mel Frequency")
        
        plt.subplot(1, 2, 2)
        plt.imshow(batch['phoneme_ids'].numpy(), aspect='auto')
        plt.title("Phoneme IDs in batch")
        plt.xlabel("Sequence Position")
        plt.ylabel("Batch Sample")
        
        plt.tight_layout()
        
        # 出力ディレクトリの作成
        output_dir = Path("test_outputs")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "preprocessing_test_batch.png")
        plt.close()
        
        print("\nVisualization saved to test_outputs/preprocessing_test_batch.png")
        
        # サンプルテキストの表示
        print("\nSample texts from batch:")
        for i, text in enumerate(batch['texts'][:3]):
            print(f"Sample {i + 1}: {text[:100]}...")  # 最初の100文字のみ表示
        
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        raise

    return batch

    #-----------------#
    """
    train_loader = data_module.train_loader
    batch = next(iter(train_loader))
    
    print("\nBatch Processing Test:")
    print(f"Batch keys: {batch.keys()}")
    print(f"Mel spectrograms shape: {batch['mel_spectrogram'].shape}")
    print(f"Phoneme IDs shape: {batch['phoneme_ids'].shape}")
    print(f"Number of stuttered samples: {batch['is_stutter'].sum().item()}")
    
    # バッチデータの可視化
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(batch['mel_spectrogram'][0].numpy(), aspect='auto')
    plt.title("First sample in batch (Mel Spectrogram)")
    
    plt.subplot(1, 2, 2)
    plt.imshow(batch['phoneme_ids'].numpy(), aspect='auto')
    plt.title("Phoneme IDs in batch")
    plt.tight_layout()
    plt.savefig("preprocessing_test_batch.png")
    plt.close()

    return batch
    """
    #-----------------------#
def test_memory_usage():
    """メモリ使用量のテスト"""
    if torch.cuda.is_available():
        print("\n=== Memory Usage ===")
        print(f"Current GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    #-----------------------#
#-----------------#
def main():
    try:
        # 設定ファイルの読み込み
        config_path = "config/config.yaml"
        print(f"\nLoading config from: {config_path}")
        
        # データモジュールの初期化
        data_module = StutterDataModule(config_path)
        
        # 単一サンプルのテスト
        audio_data, text_data = test_single_samples(data_module)
        print("\nSingle sample processing successful!")
        
        # バッチ処理のテスト
        batch = test_batch_processing(data_module)
        print("\nBatch processing successful!")
        
        # メモリ使用量のテスト
        test_memory_usage()
        
    except FileNotFoundError as e:
        print(f"\nError: Required file not found - {str(e)}")
        print("Please make sure all necessary files and directories are in place.")
    except Exception as e:
        print(f"\nError during preprocessing test: {str(e)}")
        raise
#----------------#
"""
def main():
    # 設定ファイルの読み込み
    config_path = "config/config.yaml"
    
    print(f"Testing preprocessing with config: {config_path}")
    
    # データモジュールの初期化
    data_module = StutterDataModule(config_path)
    
    # メモリ使用量の監視開始
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    try:
        # 単一サンプルのテスト
        audio_data, text_data = test_single_samples(data_module)
        print("\nSingle sample processing successful!")
        
        # バッチ処理のテスト
        batch = test_batch_processing(data_module)
        print("\nBatch processing successful!")
        
        # データセットの統計情報
        from src.data import get_dataset_stats
        stats = get_dataset_stats(data_module.config)
        print("\nDataset Statistics:")
        print(stats)
        
        # メモリ使用量の確認
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_used = (final_memory - initial_memory) / 1024 / 1024  # MB
        print(f"\nMemory used: {memory_used:.2f} MB")
        
    except Exception as e:
        print(f"Error during preprocessing test: {str(e)}")
        raise
"""
if __name__ == "__main__":
    main()