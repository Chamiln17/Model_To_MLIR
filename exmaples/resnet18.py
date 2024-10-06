import torch
from torch_mlir import fx
from torchvision.models import resnet18
from contextlib import redirect_stdout

# Define a wrapper class for the ResNet18 model
class ResNet18Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(pretrained=True)

    def forward(self, input_tensor):
        return self.model(input_tensor)

# Instantiate and prepare the model
model = ResNet18Wrapper()
model.eval()

# Create dummy input simulating an image input for ResNet18
dummy_input = torch.randn(1, 3, 224, 224)  # Adjust size as needed

# Export to MLIR using fx.export_and_import
module = fx.export_and_import(
    model,
    dummy_input,
    output_type="tosa",
    func_name=model.__class__.__name__,
)

# Save the exported MLIR module to a file
out_mlir_path = "./resnet18_linalg.mlir"
with open(out_mlir_path, 'w') as f:
    with redirect_stdout(f):
        print(module.operation.get_asm())