from typing import List, Dict, Optional
import re
from collections import defaultdict

class TextProcessor:
    def __init__(self, config):
        self.config = config
        
        # 特殊トークンの定義
        self.special_tokens = {
            "<pad>": 0,
            "<sos>": 1,
            "<eos>": 2,
            "<space>": 3,
        }
        
        if "stutter" in config and "tokens" in config["stutter"]:
            self.stutter_tokens = config["stutter"]["tokens"]
            self.special_tokens.update({
                self.stutter_tokens["repetition"]: 4,
                self.stutter_tokens["phonation"]: 5,
                self.stutter_tokens["block"]: 6,
            })
        
        # 簡単な音素マッピング（英語用）
        self.char_to_phoneme = {
            'a': 'AA',
            'b': 'B',
            'c': 'K',
            'd': 'D',
            'e': 'EH',
            'f': 'F',
            'g': 'G',
            'h': 'HH',
            'i': 'IY',
            'j': 'JH',
            'k': 'K',
            'l': 'L',
            'm': 'M',
            'n': 'N',
            'o': 'OW',
            'p': 'P',
            'q': 'K',
            'r': 'R',
            's': 'S',
            't': 'T',
            'u': 'UW',
            'v': 'V',
            'w': 'W',
            'x': 'K S',
            'y': 'Y',
            'z': 'Z',
            ' ': '<space>'
        }
        
        # 音素セットの初期化
        self.phoneme_to_id = {}
        self.id_to_phoneme = {}
        self._initialize_phoneme_set()
        
        # 単語を音素に分割する際の状態を保存
        self.word_to_phoneme_positions = defaultdict(list)

    def _initialize_phoneme_set(self):
        """音素セットの初期化"""
        # 特殊トークンの追加
        self.phoneme_to_id.update(self.special_tokens)
        
        # 基本的な音素セット（ARPAbet風）
        phonemes = [
            'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
            'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
            'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
            'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'
        ]
        
        # 音素をIDにマッピング
        for phoneme in phonemes:
            if phoneme not in self.phoneme_to_id:
                self.phoneme_to_id[phoneme] = len(self.phoneme_to_id)
                
        self.id_to_phoneme = {v: k for k, v in self.phoneme_to_id.items()}

    def clean_text(self, text: str) -> str:
        """テキストのクリーニング"""
        # 小文字化
        text = text.lower()
        # 基本的な句読点の処理
        text = re.sub(r'[^\w\s\'-]', '', text)
        # 複数のスペースを単一のスペースに
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def text_to_phonemes(self, text: str) -> Dict:
        """テキストを音素列に変換"""
        text = self.clean_text(text)
        words = text.split()
        
        phonemes = []
        word_boundaries = []
        current_pos = 0
        
        for word in words:
            word_boundaries.append(current_pos)
            word_phonemes = []
            
            # 文字ごとに音素に変換
            for char in word:
                if char.lower() in self.char_to_phoneme:
                    char_phonemes = self.char_to_phoneme[char.lower()].split()
                    word_phonemes.extend(char_phonemes)
            
            # 単語の音素位置を保存
            self.word_to_phoneme_positions[word] = list(range(current_pos, current_pos + len(word_phonemes)))
            
            phonemes.extend(word_phonemes)
            phonemes.append('<space>')
            current_pos += len(word_phonemes) + 1
        
        # 最後のスペースを削除
        if phonemes and phonemes[-1] == '<space>':
            phonemes.pop()
            current_pos -= 1
            
        return {
            "phonemes": phonemes,
            "word_boundaries": word_boundaries,
            "words": words
        }

    def phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        """音素列をIDに変換"""
        return [self.phoneme_to_id.get(p, self.phoneme_to_id['<pad>']) for p in phonemes]

    def insert_stutter_tokens(self, 
                            phoneme_ids: List[int], 
                            word_boundaries: List[int],
                            stutter_events: List[Dict]) -> List[int]:
        """吃音トークンの挿入"""
        if not stutter_events:
            return phoneme_ids
            
        new_phoneme_ids = phoneme_ids.copy()
        offset = 0
        
        for event in sorted(stutter_events, key=lambda x: x["position"]):
            word_position = event["position"]
            if word_position >= len(word_boundaries):
                continue
                
            insert_position = word_boundaries[word_position] + offset
            token_id = self.phoneme_to_id[self.stutter_tokens[event["type"]]]
            new_phoneme_ids.insert(insert_position, token_id)
            offset += 1
            
        return new_phoneme_ids

    def process_text(self, text: str, stutter_events: Optional[List[Dict]] = None) -> Dict:
        """テキストの前処理を一括で実行"""
        phoneme_data = self.text_to_phonemes(text)
        phonemes = phoneme_data["phonemes"]
        word_boundaries = phoneme_data["word_boundaries"]
        words = phoneme_data["words"]
        
        phoneme_ids = self.phonemes_to_ids(phonemes)
        
        if stutter_events:
            phoneme_ids = self.insert_stutter_tokens(
                phoneme_ids,
                word_boundaries,
                stutter_events
            )
        
        final_ids = [self.special_tokens['<sos>']] + phoneme_ids + [self.special_tokens['<eos>']]
        
        return {
            "phoneme_ids": final_ids,
            "phonemes": phonemes,
            "words": words,
            "word_boundaries": word_boundaries,
            "text": text,
            "has_stutter": bool(stutter_events)
        }

    def decode_phoneme_ids(self, phoneme_ids: List[int]) -> str:
        """音素IDを文字列に戻す（デバッグ用）"""
        return ' '.join(self.id_to_phoneme[id] for id in phoneme_ids 
                       if id in self.id_to_phoneme)

"""
import epitran
from typing import List, Dict, Optional
import re
from collections import defaultdict

class TextProcessor:
    def __init__(self, config):
        self.config = config
        self.epi = epitran.Epitran('eng-Latn')
        self.stutter_tokens = config["stutter"]["tokens"]
        
        # 特殊トークンの定義
        self.special_tokens = {
            "<pad>": 0,
            "<sos>": 1,
            "<eos>": 2,
            "<space>": 3,
            self.stutter_tokens["repetition"]: 4,
            self.stutter_tokens["phonation"]: 5,
            self.stutter_tokens["block"]: 6,
        }
        
        # 音素セットの初期化
        self.phoneme_to_id = {}
        self.id_to_phoneme = {}
        self._initialize_phoneme_set()
        
        # 単語を音素に分割する際の状態を保存
        self.word_to_phoneme_positions = defaultdict(list)

    def _initialize_phoneme_set(self):
       """#音素セットの初期化"""
"""
        # 特殊トークンの追加
        self.phoneme_to_id.update(self.special_tokens)
        self.id_to_phoneme = {v: k for k, v in self.phoneme_to_id.items()}
        
        # IPAの基本音素セット（英語）
        ipa_phonemes = [
            # 母音
            'i', 'ɪ', 'e', 'ɛ', 'æ', 'ə', 'ʌ', 'ɑ', 'ɔ', 'o', 'ʊ', 'u',
            # 二重母音
            'aɪ', 'aʊ', 'ɔɪ', 'eɪ', 'oʊ',
            # 子音
            'p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 'θ', 'ð', 's', 'z',
            'ʃ', 'ʒ', 'h', 'm', 'n', 'ŋ', 'l', 'r', 'w', 'j', 'tʃ', 'dʒ',
            # ストレスマーク
            'ˈ', 'ˌ'
        ]
        
        # 音素セットに追加
        for phoneme in ipa_phonemes:
            if phoneme not in self.phoneme_to_id:
                idx = len(self.phoneme_to_id)
                self.phoneme_to_id[phoneme] = idx
                self.id_to_phoneme[idx] = phoneme

    def clean_text(self, text: str) -> str:
        """#テキストのクリーニング
"""
        # 小文字化
        text = text.lower()
        # 基本的な句読点の処理
        text = re.sub(r'[^\w\s\'-]', '', text)
        # 複数のスペースを単一のスペースに
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def text_to_phonemes(self, text: str) -> Dict:
        """#テキストを音素列に変換し、単語と音素の位置情報も保持
"""
        text = self.clean_text(text)
        words = text.split()
        
        phonemes = []
        word_boundaries = []  # 各単語の開始位置を保存
        current_pos = 0
        
        for word in words:
            word_boundaries.append(current_pos)
            word_phonemes = list(self.epi.transliterate(word))
            
            # 単語の音素位置を保存
            self.word_to_phoneme_positions[word] = list(range(current_pos, current_pos + len(word_phonemes)))
            
            phonemes.extend(word_phonemes)
            phonemes.append('<space>')  # 単語間のスペース
            current_pos += len(word_phonemes) + 1
        
        # 最後のスペースを削除
        if phonemes and phonemes[-1] == '<space>':
            phonemes.pop()
            current_pos -= 1
            
        return {
            "phonemes": phonemes,
            "word_boundaries": word_boundaries,
            "words": words
        }

    def phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        """#音素列をIDに変換
"""
        return [self.phoneme_to_id.get(p, self.phoneme_to_id['<pad>']) for p in phonemes]

    def insert_stutter_tokens(self, 
                            phoneme_ids: List[int], 
                            word_boundaries: List[int],
                            stutter_events: List[Dict]) -> List[int]:
        """#吃音トークンの挿入
"""
        if not stutter_events:
            return phoneme_ids
            
        new_phoneme_ids = phoneme_ids.copy()
        offset = 0
        
        # イベントをソートして処理
        sorted_events = sorted(stutter_events, key=lambda x: x["position"])
        
        for event in sorted_events:
            word_position = event["position"]
            if word_position >= len(word_boundaries):
                continue
                
            # 単語の開始位置を取得
            insert_position = word_boundaries[word_position] + offset
            
            # イベントタイプに基づいてトークンを選択
            token_id = self.phoneme_to_id[self.stutter_tokens[event["type"]]]
            
            # トークンの挿入
            new_phoneme_ids.insert(insert_position, token_id)
            offset += 1
            
        return new_phoneme_ids

    def process_text(self, text: str, stutter_events: Optional[List[Dict]] = None) -> Dict:
        """#テキストの前処理を一括で実行
"""
        # テキストを音素に変換
        phoneme_data = self.text_to_phonemes(text)
        phonemes = phoneme_data["phonemes"]
        word_boundaries = phoneme_data["word_boundaries"]
        words = phoneme_data["words"]
        
        # 音素をIDに変換
        phoneme_ids = self.phonemes_to_ids(phonemes)
        
        # 吃音トークンの挿入
        if stutter_events:
            phoneme_ids = self.insert_stutter_tokens(
                phoneme_ids,
                word_boundaries,
                stutter_events
            )
        
        # 開始・終了トークンの追加
        final_ids = [self.special_tokens['<sos>']] + phoneme_ids + [self.special_tokens['<eos>']]
        
        return {
            "phoneme_ids": final_ids,
            "phonemes": phonemes,
            "words": words,
            "word_boundaries": word_boundaries,
            "text": text,
            "has_stutter": bool(stutter_events)
        }

    def decode_phoneme_ids(self, phoneme_ids: List[int]) -> str:
        """#音素IDを文字列に戻す（デバッグ用）
"""
        return ' '.join(self.id_to_phoneme[id] for id in phoneme_ids 
                       if id in self.id_to_phoneme)
"""