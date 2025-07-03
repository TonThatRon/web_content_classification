import numpy as np

# Define the mapping from domain type to one-hot vector
TYPE_MAPPING = {
    'Tổ chức': [1, 0, 0, 0, 0],
    'Báo chí, tin tức': [0, 1, 0, 0, 0],
    'Nội dung khiêu dâm': [0, 0, 1, 0, 0],
    'Cờ bạc, cá độ, vay tín dụng': [0, 0, 0, 1, 0],
    'Chưa xác định': [0, 0, 0, 0, 1],
}

def one_hot_encode(domain_type: str) -> np.ndarray:
    """Convert a domain type string into a one-hot encoded numpy array."""
    if domain_type not in TYPE_MAPPING:
        print(f"[WARNING] '{domain_type}' not in TYPE_MAPPING. Using 'Chưa xác định'")
        domain_type = 'Chưa xác định'
    return np.array(TYPE_MAPPING[domain_type], dtype=np.float32)
