"""Helper utilities to compute transcription agreement metrics."""

from __future__ import annotations

from difflib import SequenceMatcher
from itertools import combinations
from typing import Dict, Sequence, Tuple

import pandas as pd
from tqdm import tqdm


SPECIAL_TOKEN_PATTERN = ("<", ">")


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute word error rate using a simple Levenshtein distance."""

    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens:
        return float(bool(hyp_tokens))

    previous_row = list(range(len(hyp_tokens) + 1))
    for i, ref_word in enumerate(ref_tokens, start=1):
        current_row = [i]
        for j, hyp_word in enumerate(hyp_tokens, start=1):
            substitutions = previous_row[j - 1] + (ref_word != hyp_word)
            insertions = current_row[j - 1] + 1
            deletions = previous_row[j] + 1
            current_row.append(min(substitutions, insertions, deletions))
        previous_row = current_row

    return previous_row[-1] / len(ref_tokens)


def merge_transcriptions(texts: Sequence[str]) -> str:
    """Merge multiple transcripts by stitching their longest common subsequences."""

    cleaned = [text for text in texts if isinstance(text, str) and text.strip()]
    if not cleaned:
        return ""

    merged = cleaned[0]
    for text in cleaned[1:]:
        match = SequenceMatcher(None, merged, text).find_longest_match(
            0, len(merged), 0, len(text)
        )
        if match.size > 0:
            prefix = merged[: match.a]
            common = merged[match.a : match.a + match.size]
            suffix = text[match.b + match.size :]
            merged = f"{prefix}{common}{suffix}"
        else:
            merged = f"{merged} {text}"
    return merged


def pairwise_word_error_rates(texts: Sequence[str]) -> Dict[Tuple[str, str], float]:
    """Compute symmetric WER across every pair of transcripts."""

    scores: Dict[Tuple[str, str], float] = {}
    for left, right in combinations(texts, 2):
        wer = min(word_error_rate(left, right), word_error_rate(right, left))
        scores[(left, right)] = wer
    return scores


def average_word_error_rates(texts: Sequence[str]) -> Dict[str, float]:
    """Aggregate average WER scores for each transcript."""

    totals: Dict[str, Dict[str, float]] = {
        text: {"total": 0.0, "count": 0.0} for text in texts
    }
    for (left, right), score in pairwise_word_error_rates(texts).items():
        totals[left]["total"] += score
        totals[left]["count"] += 1
        totals[right]["total"] += score
        totals[right]["count"] += 1

    averages: Dict[str, float] = {}
    for text, data in totals.items():
        if data["count"]:
            averages[text] = data["total"] / data["count"]
        else:
            averages[text] = data["total"]
    return averages


def select_best_transcription(texts: Sequence[str]) -> Tuple[str, float]:
    """Return the transcript with the lowest average WER and its score.
    Therefore if there is a transcript that is identical to all others, it will be
    selected with a score of 0.0. (full match)
    """

    if not texts:
        return "", float("inf")
    averages = average_word_error_rates(texts)
    best_text = min(averages, key=averages.get)
    return best_text, averages[best_text]


def enrich_dataframe(
    df: pd.DataFrame,
    model_columns: Sequence[str],
    threshold: float,
    *,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Add agreement-related columns to the provided dataframe."""

    frame = df.copy()
    if "agreement_wrd" not in frame.columns:
        frame["agreement_wrd"] = ""
    if "best_ai_wrd" not in frame.columns:
        frame["best_ai_wrd"] = "<None>"

    iterator = frame.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=frame.shape[0])

    for idx, row in iterator:
        transcripts = [row[col] for col in model_columns if isinstance(row[col], str)]
        transcripts = [text.strip() for text in transcripts if text and text.strip()]
        transcripts = [
            text
            for text in transcripts
            if not (text.startswith(SPECIAL_TOKEN_PATTERN[0]) and text.endswith(SPECIAL_TOKEN_PATTERN[1]))
        ]
        if not transcripts:
            continue

        frame.at[idx, "agreement_wrd"] = merge_transcriptions(transcripts)
        best_text, best_score = select_best_transcription(transcripts)
        frame.at[idx, "best_ai_wrd"] = (
            best_text if best_score <= threshold else "<WER above threshold>"
        )
    return frame
