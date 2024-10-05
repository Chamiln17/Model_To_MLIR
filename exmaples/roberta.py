import torch
from torch_mlir import fx
from transformers import RobertaForMaskedLM
from contextlib import redirect_stdout

# Define a wrapper class for the RoBERTa model
class RobertaWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = RobertaForMaskedLM.from_pretrained("roberta-base", return_dict=False)

    def forward(self, input_ids):
        return self.model(input_ids)

# Instantiate and prepare the model
model = RobertaWrapper()
model.eval()

# Create dummy input simulating tokenized text input for RoBERTa
dummy_input = torch.randint(0, 50265, (1, 10))  # Adjust size as needed

# Export to MLIR using fx.export_and_import
module = fx.export_and_import(
    model,
    dummy_input,
    output_type="linalg-on-tensors",# use torch if you want to convert it into the torch Dialect
    func_name=model.__class__.__name__,
)

# Save the exported MLIR module to a file
out_mlir_path = "./roberta_linalg.mlir"
with open(out_mlir_path, 'w') as f:
    with redirect_stdout(f):
        print(module.operation.get_asm())