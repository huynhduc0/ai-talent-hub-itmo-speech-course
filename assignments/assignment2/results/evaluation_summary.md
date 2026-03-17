### Part 1 - Task 1 & 2: Greedy and Beam Baseline (LibriSpeech test-other)
- greedy_baseline: WER = 11.22%, CER = 3.81%
- beam_baseline: WER = 11.15%, CER = 3.78%

### Part 1 - Task 3: Temperature Sweep on Greedy
| Temperature | WER | CER |
|---|---|---|
| 0.5 | 11.22% | 3.81% |
| 0.8 | 11.22% | 3.81% |
| 1.0 | 11.22% | 3.81% |
| 1.2 | 11.22% | 3.81% |
| 1.5 | 11.22% | 3.81% |
| 2.0 | 11.22% | 3.81% |

### Part 2 - Task 4: Beam Search with LM Shallow Fusion
**Best Shallow Fusion Configuration:** Alpha = 1.0, Beta = 1.0 -> WER = 11.00%, CER = 3.76%

### Part 2 - Task 5: Beam Search with 4-gram LM
**4-gram LM Shallow Fusion:** Alpha = 0.05, Beta = 0.5 -> WER = 11.05%, CER = 3.77%
**4-gram LM Rescoring:** Alpha = 0.05, Beta = 0.5 -> WER = 11.12%, CER = 3.77%

### Part 2 - Task 6: Beam Search with LM Rescoring
**Best Rescoring Configuration:** Alpha = 0.01, Beta = 1.0 -> WER = 11.10%, CER = 3.77%

### Part 3 - Tasks 7 & 9: Earnings22 OOD Baseline vs LM Fallback vs Financial LM
- beam_lm_earnings_fallback: WER = 55.17%, CER = 25.44%
- beam_rescore_earnings_fallback: WER = 55.12%, CER = 25.46%
- beam_lm_earnings_financial: WER = 54.76%, CER = 25.38%
- beam_rescore_earnings_financial: WER = 55.15%, CER = 25.45%

### Part 3 - Task 7b: Temperature Sweep on Earnings22 (Greedy vs Beam SF)
| Temperature | Greedy WER | Greedy CER | Beam SF WER | Beam SF CER |
|---|---|---|---|---|
| 0.5 | 54.97% | 25.58% | 55.09% | 25.56% |
| 1.0 | 54.97% | 25.58% | 55.66% | 25.51% |
| 1.5 | 54.97% | 25.58% | 56.27% | 25.46% |
| 2.0 | 54.97% | 25.58% | 56.78% | 25.64% |

### Part 3 - Task 8: Train financial-domain KenLM
Successfully trained `lm/financial-3gram.arpa.gz` and evaluated in Tasks 7 & 9 above.
