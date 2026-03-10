import numpy as np

from causal_mask import demo_causal_mask
from cross_attention import demo_cross_attention
from inference_loop import run_inference_loop


if __name__ == "__main__":
    print("\n" + "=" * 60)
    demo_causal_mask()

    print("\n" + "=" * 60)
    demo_cross_attention()

    print("\n" + "=" * 60)
    dummy_encoder_out = np.random.randn(1, 10, 512)
    run_inference_loop(dummy_encoder_out, max_tokens=20)

    print("\n" + "=" * 60)
    print("Laboratorio concluido.")
