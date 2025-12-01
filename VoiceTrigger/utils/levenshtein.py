import math


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def is_match_by_lev(word: str, pattern: str, lv_percent: int) -> bool:
    word = word.lower()
    pattern = pattern.lower()
    dist = levenshtein_distance(word, pattern)
    max_errors = max(1, math.ceil(len(pattern) * lv_percent / 100))
    return dist <= max_errors
