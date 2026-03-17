import math
from typing import List, Tuple

import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


# ---------------------------------------------------------------------------
# Provided utility — do NOT modify
# ---------------------------------------------------------------------------

def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-100h",
            lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
            beam_width=3,
            alpha=1.0,
            beta=1.0,
            temperature=1.0,
        ):
        """
        Args:
            model_name (str): Pretrained Wav2Vec2 model from HuggingFace.
            lm_model_path (str): Path to a KenLM .arpa/.arpa.gz model.
                Pass None to disable LM (Tasks 1–3).
            beam_width (int): Number of hypotheses kept during beam search.
            alpha (float): LM weight used in shallow fusion and rescoring.
                score = log_p_acoustic + alpha * log_p_lm + beta * num_words
            beta (float): Word insertion bonus (see above).
            temperature (float): Scales acoustic logits before softmax.
                T < 1 sharpens the distribution (model more confident).
                T > 1 flattens it (model less confident, giving LM more
                influence). T = 1.0 leaves logits unchanged.
        """
        # Interact with processor/model ONLY here and in decode() to obtain
        # logits — no further model calls are allowed anywhere else.
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    # -----------------------------------------------------------------------
    # Provided utility — do NOT modify
    # -----------------------------------------------------------------------

    def _ids_to_text(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs to a decoded string."""
        text = ''.join(self.vocab[i] for i in token_ids)
        return text.replace(self.word_delimiter, ' ').strip().lower()

    # -----------------------------------------------------------------------
    # Tasks 1–4: implement the methods below
    # -----------------------------------------------------------------------

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V).

        Returns:
            str: Decoded transcript.
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        best_path = torch.argmax(log_probs, dim=-1).tolist()
        
        collapsed_path = []
        prev_token = None
        for token in best_path:
            if token != prev_token:
                if token != self.blank_token_id:
                    collapsed_path.append(token)
            prev_token = token
            
        return self._ids_to_text(collapsed_path)

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """
        Perform beam search decoding (no LM).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.
            return_beams (bool): Return all beam hypotheses for second-pass
                LM rescoring.

        Returns:
            Union[str, List[Tuple[List[int], float]]]:
                str - best decoded transcript (if return_beams=False).
                List[Tuple[List[int], float]] - list of (token_ids, log_prob)
                    tuples sorted best-first (if return_beams=True).
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        T, V = log_probs.shape
        
        # beam: {prefix_tuple: (prob_blank, prob_non_blank)}
        beam = {(): (0.0, float('-inf'))}
        
        for t in range(T):
            probs_t = log_probs[t].tolist()
            new_beam = {}
            
            for prefix, (p_b, p_nb) in beam.items():
                p_total = _log_add(p_b, p_nb)
                
                # 1. Extend with blank
                if prefix not in new_beam:
                    new_beam[prefix] = (float('-inf'), float('-inf'))
                n_p_b, n_p_nb = new_beam[prefix]
                new_beam[prefix] = (_log_add(n_p_b, p_total + probs_t[self.blank_token_id]), n_p_nb)
                
                # 2. Extend with non-blank
                for c in range(V):
                    if c == self.blank_token_id:
                        continue
                        
                    prob_c = probs_t[c]
                    extended_prefix = prefix + (c,)
                    
                    if len(prefix) > 0 and c == prefix[-1]:
                        # Extending length (requires passing through blank token)
                        if extended_prefix not in new_beam:
                            new_beam[extended_prefix] = (float('-inf'), float('-inf'))
                        e_p_b, e_p_nb = new_beam[extended_prefix]
                        new_beam[extended_prefix] = (e_p_b, _log_add(e_p_nb, p_b + prob_c))
                        
                        # Collapsing (same character, no blank token)
                        n_p_b, n_p_nb = new_beam[prefix]
                        new_beam[prefix] = (n_p_b, _log_add(n_p_nb, p_nb + prob_c))
                    else:
                        # Adding entirely new character
                        if extended_prefix not in new_beam:
                            new_beam[extended_prefix] = (float('-inf'), float('-inf'))
                        e_p_b, e_p_nb = new_beam[extended_prefix]
                        new_beam[extended_prefix] = (e_p_b, _log_add(e_p_nb, p_total + prob_c))
                        
            # Prune
            sorted_beam = sorted(new_beam.items(), key=lambda x: _log_add(x[1][0], x[1][1]), reverse=True)
            beam = dict(sorted_beam[:self.beam_width])

        res = [(list(prefix), _log_add(p_b, p_nb)) for prefix, (p_b, p_nb) in beam.items()]
        res.sort(key=lambda x: x[1], reverse=True)
        
        if return_beams:
            return res
        return self._ids_to_text(res[0][0])

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        """
        Perform beam search decoding with shallow LM fusion.

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.

        Returns:
            str: Decoded transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM shallow fusion")
        log_probs = torch.log_softmax(logits, dim=-1)
        T, V = log_probs.shape
        
        # beam: {prefix_tuple: (prob_blank, prob_non_blank)}
        beam = {(): (0.0, float('-inf'))}
        
        for t in range(T):
            probs_t = log_probs[t].tolist()
            new_beam = {}
            
            for prefix, (p_b, p_nb) in beam.items():
                p_total = _log_add(p_b, p_nb)
                
                # 1. Extend with blank
                if prefix not in new_beam:
                    new_beam[prefix] = (float('-inf'), float('-inf'))
                n_p_b, n_p_nb = new_beam[prefix]
                new_beam[prefix] = (_log_add(n_p_b, p_total + probs_t[self.blank_token_id]), n_p_nb)
                
                # 2. Extend with non-blank
                for c in range(V):
                    if c == self.blank_token_id:
                        continue
                        
                    prob_c = probs_t[c]
                    extended_prefix = prefix + (c,)
                    
                    if len(prefix) > 0 and c == prefix[-1]:
                        # Extending length (requires passing through blank)
                        if extended_prefix not in new_beam:
                            new_beam[extended_prefix] = (float('-inf'), float('-inf'))
                        e_p_b, e_p_nb = new_beam[extended_prefix]
                        new_beam[extended_prefix] = (e_p_b, _log_add(e_p_nb, p_b + prob_c))
                        
                        # Collapsing
                        n_p_b, n_p_nb = new_beam[prefix]
                        new_beam[prefix] = (n_p_b, _log_add(n_p_nb, p_nb + prob_c))
                    else:
                        if extended_prefix not in new_beam:
                            new_beam[extended_prefix] = (float('-inf'), float('-inf'))
                        e_p_b, e_p_nb = new_beam[extended_prefix]
                        new_beam[extended_prefix] = (e_p_b, _log_add(e_p_nb, p_total + prob_c))
                        
            # Prune with LM scoring
            # We must decode the token IDs to text and score with kenlm
            scored_beam = []
            for prefix, (p_b, p_nb) in new_beam.items():
                log_p_acoustic = _log_add(p_b, p_nb)
                
                # Compute LM score if there's text
                text = self._ids_to_text(prefix)
                if not text:
                    log_p_lm = 0.0
                    num_words = 0
                else:
                    # score returns log10 probabilities
                    # KenLM uses base 10 by default; multiply by ln(10) ~ 2.302585 to convert to natural log
                    # The assignment may or may not require this conversion, usually it does for alpha weighting
                    # We will follow the assignment formula: score = log_p_acoustic + alpha * log_p_lm + beta * num_words
                    # Note: KenLM score(text) returns standard log10(P). Let log_p_lm be that directly.
                    log_p_lm = self.lm_model.score(text)
                    num_words = len(text.split())
                
                total_score = log_p_acoustic + self.alpha * log_p_lm + self.beta * num_words
                scored_beam.append((prefix, new_beam[prefix], total_score))
                
            scored_beam.sort(key=lambda x: x[2], reverse=True)
            beam = {prefix: scores for prefix, scores, _ in scored_beam[:self.beam_width]}

        # Final rescored res
        final_res = []
        for prefix, (p_b, p_nb) in beam.items():
            text = self._ids_to_text(prefix)
            log_p_lm = self.lm_model.score(text) if text else 0.0
            num_words = len(text.split()) if text else 0
            
            log_p_acoustic = _log_add(p_b, p_nb)
            total_score = log_p_acoustic + self.alpha * log_p_lm + self.beta * num_words
            final_res.append((prefix, total_score))
            
        final_res.sort(key=lambda x: x[1], reverse=True)
        return self._ids_to_text(final_res[0][0])

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs.

        Args:
            beams (List[Tuple[List[int], float]]): List of (token_ids, log_prob)
                tuples from beam_search_decode(logits, return_beams=True).

        Returns:
            str: Best rescored transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")
        rescored_beams = []
        
        for token_ids, log_p_acoustic in beams:
            text = self._ids_to_text(token_ids)
            
            if not text:
                log_p_lm = 0.0
                num_words = 0
            else:
                log_p_lm = self.lm_model.score(text)
                num_words = len(text.split())
                
            total_score = log_p_acoustic + self.alpha * log_p_lm + self.beta * num_words
            rescored_beams.append((text, total_score))
            
        rescored_beams.sort(key=lambda x: x[1], reverse=True)
        return rescored_beams[0][0]

    # -----------------------------------------------------------------------
    # Provided — do NOT modify
    # -----------------------------------------------------------------------

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Run the full decoding pipeline on a raw audio tensor.

        Args:
            audio_input (torch.Tensor): 1-D or 2-D audio waveform at 16 kHz.
            method (str): One of "greedy", "beam", "beam_lm", "beam_lm_rescore".

        Returns:
            str: Decoded transcript (lowercase).
        """
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        # Temperature scaling (Task 3): flatten/sharpen the distribution
        # before log_softmax.  T=1.0 is a no-op.  Your decoders must call
        # torch.log_softmax on the logits they receive — do not call it here.
        logits = logits / self.temperature

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose one of: 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'."
            )


# ---------------------------------------------------------------------------
# Quick debug helper — run this file directly to sanity-check your decoder
# on the provided examples/ clips before evaluating on the full test sets.
# ---------------------------------------------------------------------------

def test(decoder: Wav2Vec2Decoder, audio_path: str, reference: str) -> None:
    import jiwer

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, f"Expected 16 kHz, got {sr} Hz for {audio_path}"

    print("=" * 60)
    print(f"REF : {reference}")

    for method in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        try:
            hyp = decoder.decode(audio_input, method=method)
        except NotImplementedError:
            print(f"  [{method}] not yet implemented")
            continue
        except ValueError as e:
            print(f"  [{method}] skipped ({e})")
            continue
        cer = jiwer.cer(reference, hyp)
        wer = jiwer.wer(reference, hyp)
        print(f"  [{method}] {hyp}")
        print(f"           WER={wer:.2%}  CER={cer:.2%}")


if __name__ == "__main__":
    # Reference transcripts are lowercase to match the evaluation manifests.
    # examples/ clips are for quick debugging only — use data/librispeech_test_other/
    # and data/earnings22_test/ for all reported metrics.
    test_samples = [
        ("examples/sample1.wav", "if you are generous here is a fitting opportunity for the exercise of your magnanimity if you are proud here am i your rival ready to acknowledge myself your debtor for an act of the most noble forbearance"),
        ("examples/sample2.wav", "and if any of the other cops had private rackets of their own izzy was undoubtedly the man to find it out and use the information with a beat such as that even going halves and with all the graft to the upper brackets he'd still be able to make his pile in a matter of months"),
        ("examples/sample3.wav", "guess a man gets used to anything hell maybe i can hire some bums to sit around and whoop it up when the ships come in and bill this as a real old martian den of sin"),
        ("examples/sample4.wav", "it was a tune they had all heard hundreds of times so there was no difficulty in turning out a passable imitation of it to the improvised strains of i didn't want to do it the prisoner strode forth to freedom"),
        ("examples/sample5.wav", "marguerite tired out with this long confession threw herself back on the sofa and to stifle a slight cough put up her handkerchief to her lips and from that to her eyes"),
        ("examples/sample6.wav", "at this time all participants are in a listen only mode"),
        ("examples/sample7.wav", "the increase was mainly attributable to the net increase in the average size of our fleets"),
        ("examples/sample8.wav", "operating surplus is a non cap financial measure which is defined as fully in our press release"),
    ]

    decoder = Wav2Vec2Decoder(lm_model_path=None)  # set lm_model_path for Tasks 4+

    for audio_path, reference in test_samples:
        test(decoder, audio_path, reference)
