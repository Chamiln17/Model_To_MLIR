import torch
from torch_mlir import fx
from transformers import BertForMaskedLM
from contextlib import redirect_stdout


# Define a wrapper class for the BERT model
class BertWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased", return_dict=False)

    def forward(self, input_ids):
        return self.model(input_ids)

# Instantiate and prepare the model
model = BertWrapper()
model.eval()

# Create dummy input simulating tokenized text input for BERT
# The input IDs should be in the range of the vocabulary size (0-30522 for BERT)
dummy_input = torch.randint(0, 30522, (1, 10))  # Adjust size as needed

# Export to MLIR using fx.export_and_import
module = fx.export_and_import(
    model,
    dummy_input,
    output_type="linalg-on-tensors",  # 
    func_name=model.__class__.__name__,
)

# Save the exported MLIR module to a file
out_mlir_path = "./bert_linalg.mlir"
with open(out_mlir_path, 'w') as f:
    with redirect_stdout(f):
        print(module.operation.get_asm())