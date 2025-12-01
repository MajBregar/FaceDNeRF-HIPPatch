#!/bin/bash
# Usage: ./hipify_convert.sh input.cu output.hip

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input.cu> <output.hip>"
  exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"

INCLUDE_PATH=$(python3 - <<'EOF'
import torch, os
paths = []
if hasattr(torch.utils, 'cpp_extension'):
    paths = torch.utils.cpp_extension.include_paths()
else:
    paths = [os.path.dirname(torch.__file__) + '/include']
print(paths[0])
EOF
)

hipify-clang "$INPUT_FILE" -o "$OUTPUT_FILE" \
  --cuda-path=/usr/local/cuda \
  -- \
  -std=c++17 \
  -I/usr/local/cuda/include \
  -I"$HOME/Desktop/facednerf/torch_utils/ops" \
  -I"$INCLUDE_PATH"
