import argparse
import os
import shutil
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding
from argparse import Namespace
import csv


def read_fasta(file_path: str):
    sequences: list[dict] = []
    with open(file_path, "r") as file:
        seq = []
        seq_name = None
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if not seq_name:
                    seq_name = line[1:].strip().replace(" ", "_")
                if seq:
                    sequences.append({"seq": "".join(seq), "name": seq_name})
                    seq = []
                    seq_name = line[1:].strip().replace(" ", "_")
            else:
                seq.append(line)
        if seq:
            sequences.append(
                {"seq": "".join(seq), "name": seq_name}
            )  # Add last sequence
    return sequences


def convert_outputs_to_pdb(outputs: dict) -> str:

    from transformers.models.esm.openfold_utils.protein import (
        to_pdb,
        Protein as OFProtein,
    )
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

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
            # b_factors=outputs["plddt"][i],
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))

    return "".join(pdbs)


def load_args():
    parser = argparse.ArgumentParser(description="Run ESMFold model.")
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument("--cuda", help="Use CUDA", action="store_true")
    parser.add_argument("--float16", help="Use Float16", action="store_true")
    parser.add_argument("--tf32", help="Use TensorFloat32", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=64, help="Chunk size")
    parser.add_argument("--output", required=True, help="Output directory")

    return parser.parse_args()


def get_mean_plddt_from_outputs(outputs) -> dict:
    n_pdbs = outputs["aatype"].shape[0]
    mean_plddt = {}
    for i in range(n_pdbs):
        mean_plddt[i] = outputs["plddt"][i].mean().item()
    return mean_plddt


def main(args: Namespace):

    print("Loading ESMFold tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    print("Loading ESMFold model...")
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", low_cpu_mem_usage=True
    )

    if args.cuda:
        print("Converting model to CUDA...")
        model = model.cuda()

    if args.float16:
        print("Converting model stem to float16...")
        model.esm = model.esm.half()

    if args.tf32:
        print("Using TensorFloat32.")
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Setting chunk size to {args.chunk_size}")
    model.trunk.set_chunk_size(args.chunk_size)

    # Read FASTA sequences
    print(f"Reading fastas from {args.fasta}")
    fasta_names_list = read_fasta(args.fasta)

    print(f"Tokenizing {len(fasta_names_list)} fastas...")
    fasta_list = [f["seq"] for f in fasta_names_list]
    tokenized_input = tokenizer(fasta_list, padding=False, add_special_tokens=False)[
        "input_ids"
    ]

    outputs = []
    prot = 0
    with torch.no_grad():
        for input_ids in tqdm(tokenized_input):
            prot += 1
            print(f"Modelling protein {prot} of {len(fasta_list)}...")
            try:
                input_ids = torch.tensor(
                    input_ids, device="cuda" if args.cuda else None
                ).unsqueeze(0)
                output = model(input_ids)
                outputs.append({key: val.cpu() for key, val in output.items()})
            except Exception as e:
                print(
                    f"Error occurred while processing protein {prot} of {len(fasta_list)}: {str(e)}."
                )
                print(f"Sequence: {fasta_list[prot - 1]}")
                print(f"Name: {fasta_names_list[prot - 1]['name']}")

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)

    if len(outputs) == 0:
        raise RuntimeError("No outputs generated. Check your input sequence.")

    print(f"Converting {len(outputs)} outputs to PDB files...")
    pdb_list = [convert_outputs_to_pdb(o) for o in outputs]
    for i, pdb in enumerate(pdb_list):
        with open(f"{args.output}/{fasta_names_list[i]['name']}.pdb", "w") as f:
            f.write(pdb)

    # Save pLDDT as CSV
    plddt_csv_path = os.path.join("mean_plddt.csv")
    mean_plddt = [get_mean_plddt_from_outputs(o) for o in outputs]
    with open(plddt_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "mean_pLDDT"])
        for mp in mean_plddt:
            for i, plddt_value in mp.items():
                writer.writerow([f"{fasta_names_list[i]['name']}.pdb", plddt_value])


if __name__ == "__main__":
    args = load_args()
    main(args)
