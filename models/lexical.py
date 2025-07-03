import math
import re
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from utils import get_config
config = get_config()
# Constants
PATH_DICT_SENSITIVE_WORD = config['dict_path']['dict_sensitive_word']
PATH_DICT_TLD = config['dict_path']['dict_tld_gia_re']

def normalize_domain(domain: str) -> str:
    '''
    Chuẩn hóa tên miền bằng cách loại bỏ giao thức và thay thế dấu chấm bằng khoảng trắng.
    Mục đích: đưa vào mô hình xử lý ngôn ngữ (PhoBERT).
    '''
    domain = str(domain).strip().lower()
    if domain.startswith("http://"):
        domain = domain[7:]
    elif domain.startswith("https://"):
        domain = domain[8:]
    domain = domain.replace("www.", "")
    domain = domain.replace(".", " ")
    return domain


def normalize_domain_for_lexical(domain: str) -> str:
    '''
    Chuẩn hóa tên miền cho class lexical.
    Loại bỏ http(s)://, www, và phần path sau domain chính.
    '''
    domain = str(domain).strip().lower()
    if domain.startswith("http://"):
        domain = domain[7:]
    elif domain.startswith("https://"):
        domain = domain[8:]
    if domain.startswith("www."):
        domain = domain[4:]
    domain = domain.split("/")[0]  # loại bỏ path
    return domain


def split_tld_vn(domain: str) -> str:
    '''
    Loại bỏ các TLD phổ biến của Việt Nam khỏi tên miền.
    Mục đích: giúp rút gọn domain để so sánh hoặc phân loại.
    '''
    tlds = ["edu vn", "com vn", "net vn", "org vn", "gov vn", "vn"]
    domain = normalize_domain(domain)
    for tld in tlds:
        domain = domain.replace(tld, "")
    return domain.strip()

class LexicalURLFeature:
    """A class to extract and analyze lexical features of a URL."""

    def __init__(self, url: str):
        self.url = url
        self.domain = self.extract_domain()
        self.tld = self.extract_tld()
        self.dict_word = self.load_dict_word()
        self.cheap_tlds = self.load_cheap_tld()

    def load_dict_word(self) -> Dict[str, str]:
        """Load dictionary words from an Excel file."""
        try:
            df = pd.read_excel(PATH_DICT_SENSITIVE_WORD)
            return dict(zip(df['word'], df['type']))
        except Exception as e:
            print(f"Error loading dictionary: {e}")
            return {}

    def load_cheap_tld(self) -> List[str]:
        """Load cheap TLDs from a CSV file."""
        try:
            df = pd.read_csv(PATH_DICT_TLD)
            return df.iloc[:, 0].tolist()
        except Exception as e:
            print(f"Error loading cheap TLDs: {e}")
            return []

    def extract_domain(self) -> str:
        """Extract the domain from the URL."""
        return self.url.split('.')[0]

    def extract_tld(self) -> str:
        """Extract the TLD from the URL."""
        parts = self.url.split('.', 1)
        return parts[1] if len(parts) > 1 else ''

    def get_entropy(self) -> float:
        """Calculate the entropy of the domain."""
        probs = [self.domain.count(character) / len(self.domain)
                 for character in set(self.domain)]
        entropy = -sum(p * math.log(p) / math.log(2.0) for p in probs)
        return round(entropy, 3)

    def get_length_to_feed_model(self) -> int:
        """Get the length of the domain."""
        return len(self.domain)

    def get_length_to_display(self) -> int:
        """Get the length of the full URL."""
        return len(self.url)

    def get_percentage_digits(self) -> float:
        """Calculate the percentage of digits in the domain."""
        num_digits = sum(c.isdigit() for c in self.domain)
        total_chars = len(self.domain)
        return round((num_digits / total_chars), 3) if total_chars else 0

    def get_count_special_characters(self) -> int:
        """Count the number of special characters in the domain."""
        special_chars = re.findall(
            r'[!@#$%^&*()_+\-=\[\]{};\'\\:"|<,/<>?]', self.domain)
        return len(special_chars)

    def is_cheap_tld(self) -> int:
        """Check if the TLD is in the list of cheap TLDs."""
        return int(self.tld in self.cheap_tlds)

    def get_type_url(self) -> Tuple[str, str]:
        """Determine the type of URL based on dictionary words."""
        domain = self.domain.split()[0] if " " in self.domain else self.domain
        for word, word_type in self.dict_word.items():
            if word in domain:
                return word_type, word
        return "Chưa xác định", ""


def get_vector_lexical(domain, return_all=False):
    lexical = LexicalURLFeature(domain)
    length = lexical.get_length_to_feed_model()
    entropy = lexical.get_entropy()
    percent_number = lexical.get_percentage_digits()
    number_special_char = lexical.get_count_special_characters()
    is_cheap_tld = lexical.is_cheap_tld()
    word_type, word = lexical.get_type_url()
    
    if return_all:
        return {
            "vector": [length, entropy, percent_number, number_special_char, is_cheap_tld],
            "length": length,
            "entropy": entropy,
            "percent_number": percent_number,
            "special_chars": number_special_char,
            "cheap_tld": bool(is_cheap_tld),
            "word_type": word_type,
            "word": word
        }
    else:
        return np.array([length, entropy, percent_number, number_special_char, is_cheap_tld])

