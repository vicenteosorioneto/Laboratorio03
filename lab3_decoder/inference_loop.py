import numpy as np


VOCAB_SIZE = 10000
START_TOKEN = "<START>"
EOS_TOKEN = "<EOS>"


def id_to_token(token_id: int) -> str:
    """Mapeia IDs para tokens especiais e tokens ficticios."""
    if token_id == 0:
        return START_TOKEN
    if token_id == 1:
        return EOS_TOKEN
    return f"TOKEN_{token_id}"


def generate_next_token(current_sequence, encoder_out: np.ndarray) -> np.ndarray:
    """
    Gera distribuicao de probabilidade aleatoria para o proximo token.
    """
    # encoder_out foi mantido na assinatura para simular um decoder real.
    _ = encoder_out
    _ = current_sequence

    probs = np.random.rand(VOCAB_SIZE)

    # Apos alguns passos, aumenta a probabilidade de <EOS> para facilitar a demo.
    if len(current_sequence) >= 6:
        probs[1] += 5.0

    probs = probs / np.sum(probs)
    return probs


def run_inference_loop(encoder_out: np.ndarray, max_tokens: int = 20) -> list:
    """
    Loop auto-regressivo:
    - inicia com <START>
    - escolhe proximo token por argmax
    - para em <EOS> ou max_tokens
    """
    sequence = [START_TOKEN]

    print("=== Tarefa 3: Loop de Inferencia Auto-Regressivo ===")
    print(f"Sequencia inicial: {sequence}")

    while len(sequence) < max_tokens:
        probs = generate_next_token(sequence, encoder_out)
        next_token_id = int(np.argmax(probs))
        next_token = id_to_token(next_token_id)

        sequence.append(next_token)
        print(f"Token gerado: {next_token} (id={next_token_id})")

        if next_token == EOS_TOKEN:
            print("<EOS> encontrado. Encerrando geracao.")
            break

    print("\nFrase gerada:")
    print(" ".join(sequence))

    return sequence
