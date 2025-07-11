import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np

from model_combined import PhobertLexicalContent  # Định nghĩa model ở đây
from src.feature_domain import lexical
from utils import normalize_domain, normalize_domain_for_lexical, one_hot_encode

# === Cấu hình ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEXICAL_MODEL_PATH = 'models\weight\pholexicalcontent_state_dict_30_06.pt'

# === Load model ===
phobert_config_path = 'phunganhsang/PhoBert_Lexical_Dataset55K'
content_config_path = 'RonTon05/PhoBert_content_48K'

from transformers import AutoConfig

phobert_config = AutoConfig.from_pretrained(phobert_config_path)
content_config = AutoConfig.from_pretrained(content_config_path)

model = PhobertLexicalContent(
    phobert_config=phobert_config,
    roberta_config=content_config
).to(DEVICE)

model.load_state_dict(torch.load(LEXICAL_MODEL_PATH, map_location=DEVICE))
model.eval()

# === Load tokenizers ===
tokenizer_domain = AutoTokenizer.from_pretrained(phobert_config_path)
tokenizer_content = AutoTokenizer.from_pretrained(content_config_path)

# === Hàm xử lý lexical ===
def get_lexical_features(domain: str):
    domain_norm = normalize_domain_for_lexical(domain)
    vector_lexical = lexical.get_vector_lexical(domain_norm)

    type_domain, _ = lexical.LexicalURLFeature(domain).get_type_url()
    type_mapping = {
        "bao_chi": 'Báo chí, tin tức',
        "khieu_dam": 'Nội dung khiêu dâm',
        "co_bac": 'Cờ bạc, cá độ, vay tín dụng',
        "chinh_tri": 'Tổ chức',
        "Chưa xác định": 'Chưa xác định'
    }
    mapped_type = type_mapping.get(type_domain, 'Chưa xác định')
    vector_type = one_hot_encode(mapped_type)

    vector = np.concatenate((vector_lexical, vector_type))
    return torch.tensor([vector], dtype=torch.float32).to(DEVICE)

# === Hàm phân đoạn nội dung ===
def segment_content(text, tokenizer, max_length=256, stride=128):
    tokens = tokenizer.encode(text, truncation=False)
    segments = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i + max_length]
        if len(chunk) < 10:
            continue
        decoded = tokenizer.decode(chunk, skip_special_tokens=True)
        segments.append(decoded)
    return segments

# === Pipeline test ===
def predict(domain, content):
    features = get_lexical_features(domain)

    domain_input = tokenizer_domain(
        normalize_domain(domain).split()[0],
        return_tensors="pt", padding=True, truncation=True, max_length=30
    ).to(DEVICE)

    segments = segment_content(content, tokenizer_content)
    results = []

    for segment in segments:
        content_input = tokenizer_content(
            segment, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(DEVICE)

        with torch.no_grad():
            output = model(
                features=features,
                input_ids=domain_input['input_ids'],
                attention_mask=domain_input['attention_mask'],
                input_ids_content=content_input['input_ids'],
                attention_mask_content=content_input['attention_mask'],
            )
            logits = output.logits
            probs = F.softmax(logits, dim=1)
            label = torch.argmax(probs, dim=1).item()
            results.append((label, probs.squeeze().tolist(), segment))

    return results
