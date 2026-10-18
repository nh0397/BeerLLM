# Laptop smoke test

Validated on an Intel MacBook Pro, 6-core Core i7, 32 GB RAM, Python 3.12, CPU-only PyTorch 2.2.2.

- Base: SmolLM2-360M-Instruct, pinned revision `a10cc1512eabd3dde888204e902eca88bddb4951`.
- LoRA: rank 8, query/value projections, 819,200 trainable parameters (about 0.226%).
- Two optimizer steps over eight full-guide examples completed in about 92 seconds.
- Adapter saved, loaded, and compared with the original model on one validation recipe.
- The 80-token smoke evaluation was intentionally too short to assess procedure quality. Its only purpose was verifying inference and evaluation wiring; it is not an accuracy benchmark.
- Ten unit tests pass, including pagination, missing instruction rejection, split grouping, assistant-only masking, and overlength handling.
- The full epoch uses 364 training examples from 182 recipes, plus 28 validation recipes and 49 held-out test recipes. No examples were silently truncated.

The generated guides identify unresolved brewing details and are not expert-reviewed. Training completion is not evidence of correct brewing instructions. The full-run metrics and actual answers will be recorded separately after training finishes.
