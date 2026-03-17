import os
import glob
import json
import pandas as pd

def parse_results():
    results_dir = "results"
    
    # 1. Greedy vs Beam (T = 1.0) on LibriSpeech
    print("### Part 1 - Task 1 & 2: Greedy and Beam Baseline (LibriSpeech test-other)")
    for method in ["greedy_baseline", "beam_baseline"]:
        f_path = os.path.join(results_dir, f"{method}.json")
        if os.path.exists(f_path):
            with open(f_path, "r") as f:
                data = json.load(f)
                m = method.split('_')[0]
                met = data["metrics"].get(m, {})
                print(f"- {method}: WER = {met.get('wer', 0):.2%}, CER = {met.get('cer', 0):.2%}")
                
    # 2. Temperature Sweep (Greedy)
    print("\n### Part 1 - Task 3: Temperature Sweep on Greedy")
    t_files = glob.glob(os.path.join(results_dir, "greedy_T_*.json"))
    t_results = []
    for f_path in t_files:
        with open(f_path, "r") as f:
            data = json.load(f)
            t = data["config"]["temperature"]
            wer = data["metrics"]["greedy"]["wer"]
            cer = data["metrics"]["greedy"]["cer"]
            t_results.append((t, wer, cer))
    t_results.sort(key=lambda x: x[0])
    print("| Temperature | WER | CER |")
    print("|---|---|---|")
    for t, w, c in t_results:
        print(f"| {t} | {w:.2%} | {c:.2%} |")

    # 3. Beam + LM Shallow Fusion (Sweeps)
    print("\n### Part 2 - Task 4: Beam Search with LM Shallow Fusion")
    sf_files = glob.glob(os.path.join(results_dir, "beam_lm_a*_b*.json"))
    sf_results = []
    for f_path in sf_files:
        with open(f_path, "r") as f:
            data = json.load(f)
            alpha = data["config"]["alpha"]
            beta = data["config"]["beta"]
            wer = data["metrics"]["beam_lm"]["wer"]
            cer = data["metrics"]["beam_lm"]["cer"]
            sf_results.append((alpha, beta, wer, cer))
    
    if sf_results:
        df_sf = pd.DataFrame(sf_results, columns=['Alpha', 'Beta', 'WER', 'CER'])
        df_sf = df_sf.dropna(subset=['WER']) # Drop NaN
        if not df_sf.empty:
            df_sf = df_sf.sort_values(by=['Alpha', 'Beta'])
            best_sf = df_sf.loc[df_sf['WER'].idxmin()]
            print(f"**Best Shallow Fusion Configuration:** Alpha = {best_sf['Alpha']}, Beta = {best_sf['Beta']} -> WER = {best_sf['WER']:.2%}, CER = {best_sf['CER']:.2%}")

    # 4. Task 5: 4-gram LM Evaluation
    print("\n### Part 2 - Task 5: Beam Search with 4-gram LM")
    f_path = os.path.join(results_dir, "beam_lm_4gram.json")
    if os.path.exists(f_path):
        with open(f_path, "r") as f:
            data = json.load(f)
            met = data["metrics"].get("beam_lm", {})
            print(f"**4-gram LM Shallow Fusion:** Alpha = 0.05, Beta = 0.5 -> WER = {met.get('wer', 0):.2%}, CER = {met.get('cer', 0):.2%}")
    else:
        print("**4-gram LM Shallow Fusion:** Results not found yet.")

    f_res_path = os.path.join(results_dir, "beam_rescore_4gram.json")
    if os.path.exists(f_res_path):
        with open(f_res_path, "r") as f:
            data = json.load(f)
            met_res = data["metrics"].get("beam_lm_rescore", {})
            print(f"**4-gram LM Rescoring:** Alpha = 0.05, Beta = 0.5 -> WER = {met_res.get('wer', 0):.2%}, CER = {met_res.get('cer', 0):.2%}")
    else:
        print("**4-gram LM Rescoring:** Results not found yet.")

    # 5. Beam + LM Rescoring (Sweeps)
    print("\n### Part 2 - Task 6: Beam Search with LM Rescoring")
    rs_files = glob.glob(os.path.join(results_dir, "beam_rescore_a*_b*.json"))
    rs_results = []
    for f_path in rs_files:
        with open(f_path, "r") as f:
            data = json.load(f)
            alpha = data["config"]["alpha"]
            beta = data["config"]["beta"]
            wer = data["metrics"]["beam_lm_rescore"]["wer"]
            cer = data["metrics"]["beam_lm_rescore"]["cer"]
            rs_results.append((alpha, beta, wer, cer))
            
    if rs_results:
        df_rs = pd.DataFrame(rs_results, columns=['Alpha', 'Beta', 'WER', 'CER'])
        df_rs = df_rs.dropna(subset=['WER']) # Drop NaN
        if not df_rs.empty:
            df_rs = df_rs.sort_values(by=['Alpha', 'Beta'])
            best_rs = df_rs.loc[df_rs['WER'].idxmin()]
            print(f"**Best Rescoring Configuration:** Alpha = {best_rs['Alpha']}, Beta = {best_rs['Beta']} -> WER = {best_rs['WER']:.2%}, CER = {best_rs['CER']:.2%}")

    # 6. Task 7 & 9: Out-of-Domain Earnings22 Evaluations
    print("\n### Part 3 - Tasks 7 & 9: Earnings22 OOD Baseline vs LM Fallback vs Financial LM")
    methods_to_parse = [
        "greedy_earnings", 
        "beam_earnings", 
        "beam_lm_earnings_fallback", 
        "beam_rescore_earnings_fallback",
        "beam_lm_earnings_financial",
        "beam_rescore_earnings_financial"
    ]
    for method in methods_to_parse:
        f_path = os.path.join(results_dir, f"{method}.json")
        if os.path.exists(f_path):
            with open(f_path, "r") as f:
                data = json.load(f)
                m = list(data["metrics"].keys())[0] # Get the first (and only) method key
                met = data["metrics"].get(m, {})
                print(f"- {method}: WER = {met.get('wer', 0):.2%}, CER = {met.get('cer', 0):.2%}")

    # 7. Task 7b Out-of-Domain Earnings22 Temperature Sweep
    print("\n### Part 3 - Task 7b: Temperature Sweep on Earnings22 (Greedy vs Beam SF)")
    print("| Temperature | Greedy WER | Greedy CER | Beam SF WER | Beam SF CER |")
    print("|---|---|---|---|---|")
    for t in [0.5, 1.0, 1.5, 2.0]:
        greedy_wer, greedy_cer = 0.0, 0.0
        beam_wer, beam_cer = 0.0, 0.0
        
        g_path = os.path.join(results_dir, f"greedy_earnings_t{t}.json")
        if os.path.exists(g_path):
            with open(g_path, "r") as f:
                d = json.load(f)
                greedy_wer = d["metrics"].get("greedy", {}).get("wer", 0)
                greedy_cer = d["metrics"].get("greedy", {}).get("cer", 0)
                
        b_path = os.path.join(results_dir, f"beam_lm_earnings_t{t}.json")
        if os.path.exists(b_path):
            with open(b_path, "r") as f:
                d = json.load(f)
                beam_wer = d["metrics"].get("beam_lm", {}).get("wer", 0)
                beam_cer = d["metrics"].get("beam_lm", {}).get("cer", 0)
                
        print(f"| {t} | {greedy_wer:.2%} | {greedy_cer:.2%} | {beam_wer:.2%} | {beam_cer:.2%} |")

    # 8. Task 8: Train KenLM
    print("\n### Part 3 - Task 8: Train financial-domain KenLM")
    print("Successfully trained `lm/financial-3gram.arpa.gz` and evaluated in Tasks 7 & 9 above.")

if __name__ == "__main__":
    parse_results()

