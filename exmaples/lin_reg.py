import torch
from torch_mlir import fx
from contextlib import redirect_stdout

# Define a simple linear regression model
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Instantiate and prepare the model
model = LinearRegressionModel()
model.eval()

# Create dummy input for the linear regression model
dummy_input = torch.randn(1, 1)  # Adjust size as needed

# Export to MLIR using fx.export_and_import
module = fx.export_and_import(
    model,
    dummy_input,
    output_type="torch",
    func_name=model.__class__.__name__,
)

# Save the exported MLIR module to a file
out_mlir_path = "./linear_regression_linalg.mlir"
with open(out_mlir_path, 'w') as f:
    with redirect_stdout(f):
        print(module.operation.get_asm())