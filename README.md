# About Model_To_MLIR
This project was developed in during my summer internship at the NYUAD University under the Supervision of Dr. Riyadh Baghdadi
## Project Overview
- **Project Name:** Model_To_MLIR
- **Python Version:** 3.10.12
## Setup Instructions
1. **Create Virtual Environment:**
    ```bash
    python -m venv myvenv
    ```
2. **Activate The Virtual Environment**
    ```bash
    source ./myvenv/bin/activate 
    ```
3. **Install Required Packages:**
    ```bash
    pip install -r requirements.txt
    ```

# Using The repository : 
Check `./examples` folder: 
1. In this project i used the `FX importer` since all the other importers are disabled for the time being
## Explanation of `./examples/bert.py` example
```python
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
   ```
This example demonstrates how to convert a BERT model to MLIR (Multi-Level Intermediate Representation) using the `torch_mlir` library. The script performs the following steps:

1. **Import necessary libraries**:
    - `torch`: PyTorch library for tensor computations and deep learning.
    - `torch_mlir`: Library for exporting PyTorch models to MLIR.
    - `transformers`: Hugging Face library for transformer models.
    - `contextlib.redirect_stdout`: Utility to redirect standard output.

2. **Define a wrapper class for the BERT model**:
    - `BertWrapper`: A custom PyTorch module that wraps the `BertForMaskedLM` model from the `transformers` library.

3. **Instantiate and prepare the model**:
    - Create an instance of `BertWrapper`.
    - Set the model to evaluation mode using `model.eval()`.

4. **Create dummy input**:
    - Generate a tensor simulating tokenized text input for BERT. The input IDs are randomly generated within the vocabulary size range (0-30522).

5. **Export to MLIR**:
    - Use `fx.export_and_import` to export the model to MLIR format with the specified output type (`linalg-on-tensors`) and function name.
    - There are many other dialects that are supported like `torch`, `tosa`, `stablehlo`, `raw`

6. **Save the exported MLIR module to a file**:
    - Write the MLIR representation to a file named `bert_linalg.mlir` using `redirect_stdout` to capture the output of `module.operation.get_asm()`.
   


## Remark:
Depending on the model size, the generated MLIR file size will vary. For NLP models, it can take a lot of space. For LLMs, this repo does not support them yet as it requires significant GPU and CPU resources.
<!--
This section provides resources for further understanding of the concepts discussed in the README. It includes a link to the torch-mlir architecture documentation and a reference to the torch-mlir Discord channel within the LLVM project Discord server for community support.
-->
## Ressources
1. For more in depth understanding of the concept, check the following readme file:
[touch-mlir architecture](https://github.com/llvm/torch-mlir/blob/main/docs/architecture.md)
2. Refer to the [torch-mlir](https://discord.com/channels/636084430946959380/742573221882364009) discord channel in the LLVM project discord server, you will find a lot of helpful people over there  
## Limitation Of the project :
This project is intended to help you do the conversion from PyTorch to MLIR. Although it covers a significant portion of the conversion process, it is not perfect. You need to test the generated code using the `mlir-opt` command and fix any syntax error by yourself .