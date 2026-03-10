# Laboratorio 03 - Transformer Decoder (NumPy)

Implementacao didatica de partes fundamentais de um **Transformer Decoder** usando apenas **Python + NumPy**.

## Objetivo

Este laboratorio demonstra, de forma simples e direta, tres componentes importantes:

1. **Mascara Causal (Look-Ahead Mask)**
2. **Cross Attention (Encoder -> Decoder)**
3. **Loop de Inferencia Auto-Regressivo**

Nao ha foco em otimizacao de performance. A ideia e entender o fluxo matematico e os formatos dos tensores.

## Estrutura do projeto

```text
lab3_decoder/
|-- main.py
|-- causal_mask.py
|-- cross_attention.py
`-- inference_loop.py
```

## Requisitos

- Python 3.9+
- NumPy

Instalacao do NumPy:

```bash
pip install numpy
```

## Como executar

A partir da raiz do repositorio:

```bash
cd lab3_decoder
python main.py
```

## O que cada arquivo faz

### `lab3_decoder/causal_mask.py`

- Implementa `create_causal_mask(seq_len)`:
  - diagonal principal e triangular inferior = `0`
  - triangular superior = `-np.inf`
- Gera `Q` e `K` aleatorios.
- Calcula:
  - `scores = (Q @ K.T) / sqrt(d_k)`
  - `scores_mascarados = scores + mascara`
  - `probs = softmax(scores_mascarados)`
- Imprime no console:
  - scores antes da mascara
  - scores depois da mascara
  - probabilidades finais
  - confirmacao de que posicoes futuras tem probabilidade `0.0`

### `lab3_decoder/cross_attention.py`

- Simula tensores com shapes:
  - `encoder_output`: `[1, 10, 512]`
  - `decoder_state`: `[1, 4, 512]`
- Implementa `cross_attention(encoder_out, decoder_state)` com pesos `Wq`, `Wk`, `Wv`.
- Calcula:
  - `Q = decoder_state @ Wq`
  - `K = encoder_out @ Wk`
  - `V = encoder_out @ Wv`
  - `scores = (Q @ K.transpose(0,2,1)) / sqrt(d_model)`
  - `attn_weights = softmax(scores)`
  - `output = attn_weights @ V`
- Mostra o shape final do `output`.

### `lab3_decoder/inference_loop.py`

- Define vocabulario ficticio de tamanho `10000`.
- Implementa `generate_next_token(current_sequence, encoder_out)`:
  - cria probabilidades aleatorias
  - normaliza para somar `1`
- Implementa loop auto-regressivo:
  - inicia com `['<START>']`
  - escolhe proximo token por `argmax`
  - para em `<EOS>` ou ao atingir `20` tokens
- Imprime a frase final gerada.

## Resultado esperado (resumo)

Ao executar `main.py`, voce deve ver:

1. **Tarefa 1** com a mascara causal aplicada corretamente e probabilidades futuras zeradas.
2. **Tarefa 2** com `output.shape = (1, 4, 512)`.
3. **Tarefa 3** gerando tokens ate encontrar `<EOS>` ou atingir o limite maximo.

## Observacao

Este projeto e educacional. Em implementacoes reais de Transformers, costuma-se usar frameworks como PyTorch/TensorFlow, multi-head attention, batching mais robusto e tecnicas de estabilidade numerica adicionais.
