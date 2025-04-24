import numpy as np

def dcg_at_k(retrieved, target, k):
    """
    Compute DCG@k (Discounted Cumulative Gain).
    """
    retrieved = retrieved[:k]
    gains = [1.0 if item == target else 0.0 for item in retrieved]
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return np.sum(gains / discounts)

def ndcg_at_k(retrieved, target, k):
    """
    Compute NDCG@k.
    """
    dcg = dcg_at_k(retrieved, target, k)
    ideal_dcg = dcg_at_k([target], target, k)  # Ideal DCG: only the target at top
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0