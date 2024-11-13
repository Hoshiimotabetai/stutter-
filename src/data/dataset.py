import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional
from .audio_processor import AudioProcessor
from .text_processor import TextProcessor

class LibriSpeechDataset(Dataset):
    """LibriSpeech dataset implementation"""
    def __init__(self, config: Dict, split: str = "train"):
        self.config = config
        self.split = split
        self.audio_processor = AudioProcessor(config)
        self.text_processor = TextProcessor(config)
        
        # Load data
        self.data = self._load_data()
        print(f"[{split}] Loaded {len(self.data)} utterances")

    def _load_data(self) -> List[Dict]:
        """Load dataset files"""
        data = []
        librispeech_path = Path(self.config["data"]["librispeech_path"])
        
        # Set proper split directory
        if self.split == "train":
            subsets = ["train-clean-100"]
        elif self.split == "valid":
            subsets = ["dev-clean"]
        else:  # test
            subsets = ["test-clean"]
        
        # Load all transcript files
        for subset in subsets:
            subset_path = librispeech_path / subset
            if not subset_path.exists():
                print(f"Warning: {subset_path} does not exist")
                continue
            
            # Find all transcript files
            for trans_file in subset_path.glob("**/*.trans.txt"):
                # Read transcripts
                with open(trans_file, "r", encoding="utf-8") as f:
                    for line in f:
                        file_id, text = line.strip().split(" ", 1)
                        # Find corresponding audio file
                        wav_path = trans_file.parent / f"{file_id}.flac"
                        
                        if wav_path.exists():
                            data.append({
                                "id": file_id,
                                "speaker_id": trans_file.parent.parent.name,
                                "text": text,
                                "audio_path": str(wav_path)
                            })
        
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single item from the dataset"""
        item = self.data[idx]
        
        # Process audio
        audio_data = self.audio_processor.process_audio(
            item["audio_path"],
            max_duration=self.config["data"].get("max_duration")
        )
        
        # Process text
        text_data = self.text_processor.process_text(item["text"])
        
        return {
            "id": item["id"],
            "speaker_id": item["speaker_id"],
            "text": item["text"],
            "phoneme_ids": text_data["phoneme_ids"],
            "mel_spectrogram": audio_data["mel_spectrogram"],
            "waveform": audio_data["waveform"],
            "duration": audio_data["duration"],
            "mel_masks": torch.ones(audio_data["mel_spectrogram"].size(2))  # Time dimension mask
        }

def collate_batch(batch: List[Dict]) -> Dict:
    """Collate batch items"""
    # Get maximum lengths
    max_mel_len = max(x["mel_spectrogram"].size(2) for x in batch)
    max_phoneme_len = max(len(x["phoneme_ids"]) for x in batch)
    
    # Prepare tensors
    batch_size = len(batch)
    mel_dim = batch[0]["mel_spectrogram"].size(1)
    
    # Initialize padded tensors
    mel_padded = torch.zeros(batch_size, mel_dim, max_mel_len)
    phoneme_padded = torch.zeros(batch_size, max_phoneme_len, dtype=torch.long)
    mel_masks = torch.zeros(batch_size, max_mel_len)
    phoneme_masks = torch.zeros(batch_size, max_phoneme_len)
    
    # Metadata lists
    ids = []
    speakers = []
    texts = []
    
    # Fill padded tensors
    for i, item in enumerate(batch):
        mel = item["mel_spectrogram"]
        mel_len = mel.size(2)
        phoneme = torch.tensor(item["phoneme_ids"])
        phoneme_len = len(phoneme)
        
        mel_padded[i, :, :mel_len] = mel
        phoneme_padded[i, :phoneme_len] = phoneme
        mel_masks[i, :mel_len] = 1
        phoneme_masks[i, :phoneme_len] = 1
        
        ids.append(item["id"])
        speakers.append(item["speaker_id"])
        texts.append(item["text"])
    
    return {
        "ids": ids,
        "speaker_ids": speakers,
        "texts": texts,
        "phoneme_ids": phoneme_padded,
        "mel_spectrogram": mel_padded,
        "mel_masks": mel_masks,
        "phoneme_masks": phoneme_masks
    }