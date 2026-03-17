import os
import json
import argparse
import pandas as pd
import torch
import torchaudio
import jiwer
from tqdm import tqdm
from wav2vec2decoder import Wav2Vec2Decoder

def run_evaluation(
    dataset_path: str,
    output_path: str,
    methods: list,
    model_name: str = "facebook/wav2vec2-base-100h",
    lm_model_path: str = "lm/3-gram.pruned.1e-7.arpa.gz",
    beam_width: int = 3,
    alpha: float = 1.0,
    beta: float = 1.0,
    temperature: float = 1.0
):
    """
    Evaluates the Wav2Vec2Decoder on a dataset and saves results to JSON.
    """
    print(f"Loading dataset from: {dataset_path}")
    manifest_path = os.path.join(dataset_path, "manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Load dataset
    df = pd.read_csv(manifest_path)
    # The dataset path might be relative to assignment root in manifest
    # Make sure we use absolute paths if needed or just trust the manifest if we run from root
    
    decoder = Wav2Vec2Decoder(
        model_name=model_name,
        lm_model_path=lm_model_path,
        beam_width=beam_width,
        alpha=alpha,
        beta=beta,
        temperature=temperature
    )

    results = {
        "config": {
            "dataset": dataset_path,
            "model_name": model_name,
            "lm_model_path": lm_model_path,
            "beam_width": beam_width,
            "alpha": alpha,
            "beta": beta,
            "temperature": temperature,
            "methods": methods
        },
        "metrics": {},
        "samples": []
    }

    print(f"Running evaluation for methods: {methods}")
    
    sample_results = {method: [] for method in methods}
    references = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Assuming the path in manifest is relative to the working directory (assignment2)
        audio_path = row['path']
        reference = str(row['text']).lower().strip()
        
        try:
            audio_input, sr = torchaudio.load(audio_path)
            if sr != 16000:
                # Optionally resample, but prompt says it's 16kHz
                pass
            
            sample_data = {"id": idx, "audio": audio_path, "reference": reference, "hypotheses": {}}
            references.append(reference)

            for method in methods:
                try:
                    hyp = decoder.decode(audio_input, method=method)
                    sample_data["hypotheses"][method] = hyp
                    sample_results[method].append(hyp)
                except NotImplementedError:
                    sample_data["hypotheses"][method] = "<NOT_IMPLEMENTED>"
                    sample_results[method].append("")
                except Exception as e:
                    sample_data["hypotheses"][method] = f"<ERROR: {e}>"
                    sample_results[method].append("")

            results["samples"].append(sample_data)

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    # Compute aggregate metrics
    for method in methods:
        hyps = sample_results[method]
        # Filter out errors for metric computation if any
        valid_refs = []
        valid_hyps = []
        for r, h in zip(references, hyps):
            if not h.startswith("<") and h != "":  # basic check to ignore errors
                valid_refs.append(r)
                valid_hyps.append(h)
        
        if valid_refs:
            wer = jiwer.wer(valid_refs, valid_hyps)
            cer = jiwer.cer(valid_refs, valid_hyps)
            results["metrics"][method] = {"wer": wer, "cer": cer}
            print(f"Method: {method} | WER: {wer:.2%} | CER: {cer:.2%}")
        else:
            results["metrics"][method] = {"wer": None, "cer": None}

    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Results saved to {output_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to save JSON results")
    parser.add_argument("--methods", type=str, nargs='+', default=["greedy", "beam"], help="Methods to evaluate")
    parser.add_argument("--beam_width", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lm_path", type=str, default="lm/3-gram.pruned.1e-7.arpa.gz")
    
    args = parser.parse_args()
    
    # Disable LM if only evaluating non-LM methods to save loading time
    if all(m in ["greedy", "beam"] for m in args.methods):
        args.lm_path = None
        
    run_evaluation(
        dataset_path=args.dataset,
        output_path=args.output,
        methods=args.methods,
        beam_width=args.beam_width,
        alpha=args.alpha,
        beta=args.beta,
        temperature=args.temperature,
        lm_model_path=args.lm_path
    )
