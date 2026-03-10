import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax estavel numericamente."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def cross_attention(encoder_out: np.ndarray, decoder_state: np.ndarray) -> np.ndarray:
    """
    Cross-attention (Encoder -> Decoder):
    Q vem do decoder e K/V vem do encoder.
    """
    d_model = encoder_out.shape[-1]

    # Pesos simples para projecoes lineares.
    Wq = np.random.randn(d_model, d_model) / np.sqrt(d_model)
    Wk = np.random.randn(d_model, d_model) / np.sqrt(d_model)
    Wv = np.random.randn(d_model, d_model) / np.sqrt(d_model)

    Q = decoder_state @ Wq
    K = encoder_out @ Wk
    V = encoder_out @ Wv

    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_model)
    attn_weights = softmax(scores, axis=-1)
    output = attn_weights @ V

    return output


def demo_cross_attention() -> None:
    """Demonstra a tarefa de cross-attention com shapes solicitados."""
    np.random.seed(7)

    encoder_output = np.random.randn(1, 10, 512)
    decoder_state = np.random.randn(1, 4, 512)

    output = cross_attention(encoder_output, decoder_state)

    print("=== Tarefa 2: Cross Attention ===")
    print(f"Shape de encoder_output: {encoder_output.shape}")
    print(f"Shape de decoder_state: {decoder_state.shape}")
    print(f"Shape de output: {output.shape}")
