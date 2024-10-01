
import mlx.core as mlx
from mlx.models import phi

# Load the Phi-3.5 Mini model
model = phi.Phi("phi-3.5-mini")
# Prepare input
input_text = "Explain the concept of quantum entanglement."
# Run inference
output = model.generate(input_text, max_tokens=100)
# Print the result
print(output)