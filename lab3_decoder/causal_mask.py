import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax estavel numericamente."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Cria mascara causal [seq_len, seq_len]:
    - diagonal + triangular inferior = 0
    - triangular superior = -inf
    """
    mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
    return mask


def demo_causal_mask() -> None:
    """Demonstra o uso da mascara causal no scaled dot-product attention."""
    np.random.seed(42)

    seq_len = 5
    d_k = 4

    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)

    scores = (Q @ K.T) / np.sqrt(d_k)
    mask = create_causal_mask(seq_len)
    masked_scores = scores + mask
    probs = softmax(masked_scores, axis=-1)

    np.set_printoptions(precision=4, suppress=True)

    print("=== Tarefa 1: Mascara Causal ===")
    print("\nScores antes da mascara:")
    print(scores)

    print("\nScores depois da mascara:")
    print(masked_scores)

    print("\nProbabilidades finais (softmax):")
    print(probs)

    # Mostra explicitamente as probabilidades em posicoes futuras (j > i).
    future_probs = probs[np.triu_indices(seq_len, k=1)]
    print("\nProbabilidades em posicoes futuras (deve ser 0.0):")
    print(future_probs)

    all_future_zero = np.all(future_probs == 0.0)
    print(f"\nTodas as posicoes futuras sao 0.0? {all_future_zero}")
