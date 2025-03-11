import os
import subprocess
from HorusAPI import PluginBlock, PluginVariable, VariableTypes, Extensions

# Inputs
fasta_input = PluginVariable(
    id="fasta_input",
    name="Fasta Input",
    description="The input fasta file containing the protein sequences.",
    type=VariableTypes.FILE,
    allowedValues=["fasta"],
)

# Variables
use_cuda = PluginVariable(
    id="use_cuda",
    name="Use CUDA",
    description="Whether to use CUDA for inference.",
    type=VariableTypes.BOOLEAN,
    defaultValue=False,
)

float16 = PluginVariable(
    id="float16",
    name="Use float16",
    description="Convert the language model stem to float16 to improve performance and memory usage",
    type=VariableTypes.BOOLEAN,
    defaultValue=True,
)

tensorfloat_32 = PluginVariable(
    id="tensorfloat_32",
    name="TensorFloat32",
    description="Enable TensorFloat32 computation for a general speedup if your hardware supports it. This line has no effect if your hardware doesn't support it.",
    type=VariableTypes.BOOLEAN,
    defaultValue=True,
)

chunk_size = PluginVariable(
    id="chunk_size",
    name="Chunk Size",
    description="'chunk_size' used in the folding trunk. Smaller chunk sizes use less memory, but have slightly worse performance.",
    type=VariableTypes.INTEGER,
    defaultValue=64,
)

# Outputs
output_directory = PluginVariable(
    id="output_directory",
    name="Output Directory",
    description="The directory where the predicted 3D structures will be saved.",
    type=VariableTypes.FOLDER,
)

SUBMIT_ESMFOLD_PATH = os.path.abspath(
    os.path.join("Include", "Scripts", "submit_esmfold.py")
)


def fold_proteins(block: PluginBlock):

    # Extract necessary values from the block
    use_cuda_value = block.variables[use_cuda.id]
    float16_value = block.variables[float16.id]
    tensorfloat_32_value = block.variables[tensorfloat_32.id]
    chunk_size_value = block.variables[chunk_size.id]
    input_value = block.inputs[fasta_input.id]

    # Call the new script
    output_folder = "esm_fold_outputs"

    # Call the python script directly wth exec
    run_in_horus = block.config["run_within_horus"]

    if run_in_horus:
        from argparse import Namespace
        from Scripts.submit_esmfold import main as submit_esmfold

        submit_esmfold(
            Namespace(
                **{
                    "cuda": use_cuda_value,
                    "float16": float16_value,
                    "tf32": tensorfloat_32_value,
                    "chunk_size": chunk_size_value,
                    "output": output_folder,
                    "fasta": input_value,
                }
            )
        )

    else:

        command: list[str] = str(block.config["run_command_config"]).split(" ") + [
            SUBMIT_ESMFOLD_PATH
        ]

        options = [
            "--fasta",
            input_value,
            "--output",
            output_folder,
            "--cuda" if use_cuda_value else None,
            "--float16" if float16_value else None,
            "--tf32" if tensorfloat_32_value else None,
        ]

        options = [o for o in options if o]

        print("Running ESMFold from external command")
        with subprocess.Popen(
            command + options,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as p:
            if p.stdout:
                for line in p.stdout:
                    print(line)

            if p.stderr:
                for line in p.stderr:
                    print(line)

            p.wait()

            if p.returncode != 0:
                raise RuntimeError(f"Error running ESMFold")

    computed_plddt = "mean_plddt.csv"
    if os.path.exists(computed_plddt):
        Extensions().loadCSV(computed_plddt, title="Mean pLDDT")

    block.setOutput(output_directory.id, output_folder)


esmfold_block = PluginBlock(
    id="esmfold",
    name="ESMFold",
    description="Predict 3D structures from protein sequences using ESMFold.",
    inputs=[fasta_input],
    variables=[use_cuda, float16, tensorfloat_32, chunk_size],
    action=fold_proteins,
    outputs=[output_directory],
)
