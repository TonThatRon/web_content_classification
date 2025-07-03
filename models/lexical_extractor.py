import torch
import numpy as np
from .lexical import LexicalURLFeature, normalize_domain_for_lexical, get_vector_lexical,normalize_domain,split_tld_vn
from .onehot import one_hot_encode


def get_combined_lexical_vector(domain):
    domain_clean = normalize_domain_for_lexical(domain)
    lexical_vector = np.array(get_vector_lexical(domain_clean))  # đảm bảo là np.ndarray

    type_domain, _ = LexicalURLFeature(domain).get_type_url()
    type_mapping = {
        "bao_chi": "Báo chí, tin tức",
        "khieu_dam": "Nội dung khiêu dâm",
        "co_bac": "Cờ bạc, cá độ, vay tín dụng",
        "chinh_tri": "Tổ chức",
        "Chưa xác định": "Chưa xác định"
    }
    domain_type_label = type_mapping.get(type_domain, "Chưa xác định")
    onehot_vector = np.array(one_hot_encode(domain_type_label))  # đảm bảo cũng là np.ndarray

    combined_vector = np.concatenate([lexical_vector, onehot_vector])  # ghép lại thành 10 chiều
    return torch.tensor([combined_vector], dtype=torch.float32)



print(get_combined_lexical_vector("ronton.id.vn"))