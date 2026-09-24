CUDA_MM="$(
  nvcc --version \
    | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' \
    | head -1
)"

case "$CUDA_MM" in
  12.8)
    TORCH_BACKEND="cu128"
    ;;
  12.9)
    TORCH_BACKEND="cu129"
    ;;
  13.0)
    TORCH_BACKEND="cu130"
    ;;
  *)
    echo "Unsupported or unverified CUDA toolkit version: ${CUDA_MM}"
    echo "Use a Lambda image with CUDA 12.8, 12.9, or 13.0."
    exit 1
    ;;
esac

echo "CUDA toolkit: ${CUDA_MM}"
echo "vLLM/PyTorch backend: ${TORCH_BACKEND}"