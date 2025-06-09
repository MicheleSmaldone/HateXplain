#!/usr/bin/env bash
set -euo pipefail

# run LIME on the supervised‐attention model
echo "→ Running LIME for bert_supervised…"
python testing_with_lime.py bert_supervised 400 0.001

# run LIME on the sparsemax‐attention model
echo "→ Running LIME for bert_sparsemax…"
python testing_with_lime.py bert_sparsemax 400 0.001

echo "✅ All done!"
