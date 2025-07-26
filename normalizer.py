import re
from parsnorm import ParsNorm
import pandas as pd

def map_words_dict(csv_file="map_words.csv"):
    map_df = pd.DataFrame(pd.read_csv(csv_file))
    mapping_dict = {}
    for i in range(len(map_df['original'])):
        mapping_dict[map_df['original'][i]] = map_df['corrected'][i]
    return {str(k): str(v) for k, v in mapping_dict.items() if pd.notna(k) and pd.notna(v)}

def create_pattern_from_mapping_dict(mapping_dict):
    # Use \b (word boundary) to ensure only full words are matched
    return r"\b(" + "|".join(map(re.escape, mapping_dict.keys())) + r")\b"

def multiple_replace(text, mapping_dict, mapping_pattern):
    return re.sub(mapping_pattern, lambda m: mapping_dict[m.group()], str(text))

class Normalizer:
    def __init__(self, map_words_path='./final_map_words.csv'):
        self.words_mapping_dict = map_words_dict(map_words_path)
        self.words_mapping_pattern = create_pattern_from_mapping_dict(self.words_mapping_dict)
        self.norm_obj = ParsNorm(allowed_puncts=".،!؟:?؛", remove_diacritics=False)
    
    def normalize(self, text):
        text = self.norm_obj.normalize(text, remove_punct=False)
        text = multiple_replace(text, self.words_mapping_dict, self.words_mapping_pattern)
        text = text.replace("?", "؟")
        text = text.replace("؛", ".")
        text = text.replace("ۀ", "ه‌ی")
        text = re.sub(r"\s+([.!؟،:])", r"\1", text)
        text = re.sub(r"\s+", " ", text)
        return text
    