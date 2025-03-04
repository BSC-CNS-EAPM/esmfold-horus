"""
Run ESMFold model from huggingface transformers library

"""

import argparse as ap
import os
from typing import List

import torch


class ESMFold:
    def __init__(
        self, output_folder, reduce_chunk_size=True, esmfold_path="facebook/esmfold_v1"
    ):
        self._validate_inputs(esmfold_path, output_folder, reduce_chunk_size)
        self.esmfold_path = esmfold_path
        self.output_folder = output_folder
        self.reduce_chunk_size = reduce_chunk_size
        # Initializes self.model and self.tokenizer
        self._init_model()

    def _validate_inputs(
        self,
        esmfold_path: str,
        output_folder: str,
        reduce_chunk_size: bool = False,
    ):
        if not os.path.exists(esmfold_path) and esmfold_path != "facebook/esmfold_v1":
            raise FileNotFoundError(f"Params path {esmfold_path} does not exist.")

        if not os.path.exists(output_folder):
            print(f"Creating output folder {output_folder}")
            os.makedirs(output_folder, exist_ok=True)
            # raise FileNotFoundError(f"Folder path {output_folder} does not exist.")

        if not isinstance(reduce_chunk_size, bool):
            raise ValueError(
                f"reduce_chunck_size must be a boolean. "
                f"Received: {reduce_chunk_size}"
            )

    def _init_model(self):
        from transformers import AutoTokenizer, EsmForProteinFolding

        self.tokenizer = AutoTokenizer.from_pretrained(self.esmfold_path)
        model = EsmForProteinFolding.from_pretrained(
            self.esmfold_path, low_cpu_mem_usage=True
        )

        if torch.cuda.is_available():
            model = model.cuda()

        ### Optimize model performance ###
        # Convert the language model stem to float16 to improve performance and
        # memory usage when running on a modern GPU. This was used during model
        # training, and so should not make the outputs from the rest of the
        # model invalid.
        model.esm = model.esm.half()

        # Enable TensorFloat32 computation for a general speedup if your hardware
        # supports it. This line has no effect if your hardware doesn't support it.
        torch.backends.cuda.matmul.allow_tf32 = True

        # Use this if your GPU memory is 16GB or less, or if you're folding
        # longer (over 600 or so) sequences. Smaller chunk sizes use less
        # memory, but have slightly worse performance
        if self.reduce_chunk_size:
            model.trunk.set_chunk_size(64)

        self.model = model

    def get_mean_plddt_from_outputs(outputs) -> dict:
        n_pdbs = outputs["aatype"].shape[0]
        mean_plddt = {}
        for i in range(n_pdbs):
            mean_plddt[i] = outputs["plddt"][i].mean().item()
        return mean_plddt

    def convert_outputs_to_pdb(
        self,
        n_pdbs,  # dict EsmForProteinFoldingOutput  object
        output_dir: str = "esmfold",
        save_pdbs: bool = False,  # If True, save PDB files
        out_filename: str = "decoy",
    ):
        from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
        from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
        from transformers.models.esm.openfold_utils.protein import to_pdb

        # Check inputs
        if not isinstance(save_pdbs, bool):
            raise ValueError(f"save_pdbs must be a boolean. " f"Received: {save_pdbs}")
        pdbs_dict = {}
        for y, outputs in enumerate(n_pdbs):
            final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
            outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
            final_atom_positions = final_atom_positions.cpu().numpy()
            final_atom_mask = outputs["atom37_atom_exists"]
            pdbs = []
            for i in range(outputs["aatype"].shape[0]):
                aa = outputs["aatype"][i]
                pred_pos = final_atom_positions[i]
                mask = final_atom_mask[i]
                resid = outputs["residue_index"][i] + 1
                pred = OFProtein(
                    aatype=aa,
                    atom_positions=pred_pos,
                    atom_mask=mask,
                    residue_index=resid,
                    b_factors=outputs["plddt"][i],
                    chain_index=(
                        outputs["chain_index"][i] if "chain_index" in outputs else None
                    ),
                )
                pdbs.append(to_pdb(pred))

                if save_pdbs:
                    with open(
                        os.path.join(output_dir, f"{out_filename}_{y}_{i}.pdb"),
                        "w",
                    ) as f:
                        f.write(to_pdb(pred))
            pdbs_dict[y] = pdbs
        return pdbs_dict

    def predict(
        self,
        sequences: List[str],
        save_pdb: bool = False,
        save_plddt: bool = False,
        out_filename: str = "decoy",
    ):
        import json

        from tqdm import tqdm

        # Now let's pull out the sequences and batch-tokenize all of them.
        tokenized_input = self.tokenizer(
            sequences, padding=False, add_special_tokens=False
        )["input_ids"]

        # Only GPU is supported
        if torch.cuda.is_available():
            device = "cuda"
            tokenized_input = tokenized_input.cuda()
        else:
            print("No GPU available. Using CPU.")
            device = "cpu"

        outputs = []
        with torch.no_grad():
            for input_ids in tqdm(tokenized_input):
                input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
                output = self.model(input_ids)
                outputs.append({key: val.cpu() for key, val in output.items()})

        if save_pdb:
            self.convert_outputs_to_pdb(
                outputs,
                save_pdbs=save_pdb,
                output_dir=self.output_folder,
                out_filename=out_filename,
            )
        if save_plddt:
            mean_plddt = self.get_mean_plddt_from_outputs(outputs)
            mean_plddt_with_filename = {
                f"{out_filename}_{i}.pdb": mean_plddt[i] for i in mean_plddt.keys()
            }
            with open(
                os.path.join(self.output_folder, f"{out_filename}_plddt.json"), "w"
            ) as f:
                json.dump(mean_plddt_with_filename, f)

        return outputs
