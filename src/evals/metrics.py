import math


def recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """
    Computes Recall@K: the proportion of expected documents found in the top K retrieved documents.
    """
    if not expected_ids:
        return 0.0
    
    retrieved_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in expected_ids if doc_id in retrieved_k)
    return hits / len(expected_ids)


def mrr_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """
    Computes Mean Reciprocal Rank@K: 1 / rank of the first relevant document.
    """
    if not expected_ids:
        return 0.0

    retrieved_k = retrieved_ids[:k]
    for rank, doc_id in enumerate(retrieved_k, start=1):
        if doc_id in expected_ids:
            return 1.0 / rank
            
    return 0.0


def dcg_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """
    Computes Discounted Cumulative Gain at K.
    """
    retrieved_k = retrieved_ids[:k]
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_k):
        if doc_id in expected_ids:
            # Relevance score is binary (1 if expected, 0 otherwise)
            dcg += 1.0 / math.log2(i + 2) # i is 0-indexed, we want log2(rank + 1) -> log2(i + 2)
    return dcg


def ndcg_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at K.
    """
    if not expected_ids:
        return 0.0
        
    actual_dcg = dcg_at_k(retrieved_ids, expected_ids, k)
    
    # Ideal DCG: what if all expected documents were ranked at the top?
    ideal_retrieved = expected_ids[:k] 
    ideal_dcg = dcg_at_k(ideal_retrieved, expected_ids, k)
    
    if ideal_dcg == 0.0:
        return 0.0
        
    return actual_dcg / ideal_dcg
