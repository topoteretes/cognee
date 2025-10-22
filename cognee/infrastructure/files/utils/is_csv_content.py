import csv
from collections import Counter


def is_csv_content(content):
    """
    Heuristically determine whether a bytes-like object is CSV text.

    Strategy (fail-fast and cheap to expensive):
      1) Decode: Try a small ordered list of common encodings with strict errors.
      2) Line sampling: require >= 2 non-empty lines; sample up to 50 lines.
      3) Delimiter detection:
         - Prefer csv.Sniffer() with common delimiters.
         - Fallback to a lightweight consistency heuristic.
      4) Lightweight parse check:
         - Parse a few lines with the delimiter.
         - Ensure at least 2 valid rows and relatively stable column counts.

    Returns:
        bool: True if the buffer looks like CSV; False otherwise.
    """
    try:
        encoding_list = [
            "utf-8",
            "utf-8-sig",
            "utf-32-le",
            "utf-32-be",
            "utf-16-le",
            "utf-16-be",
            "gb18030",
            "shift_jis",
            "cp949",
            "cp1252",
            "iso-8859-1",
        ]

        # Try to decode strictly—if decoding fails for all encodings, it's not text/CSV.
        text = None
        for enc in encoding_list:
            try:
                text = content.decode(enc, errors="strict")
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            return False

        # Reject empty/whitespace-only payloads.
        stripped = text.strip()
        if not stripped:
            return False

        # Split into logical lines and drop empty ones. Require at least two lines.
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            return False

        # Take a small sample to keep sniffing cheap and predictable.
        sample_lines = lines[:50]

        # Detect delimiter using csv.Sniffer first; if that fails, use our heuristic.
        delimiter = _sniff_delimiter(sample_lines) or _heuristic_delimiter(sample_lines)
        if not delimiter:
            return False

        # Finally, do a lightweight parse sanity check with the chosen delimiter.
        return _lightweight_parse_check(sample_lines, delimiter)
    except Exception:
        return False


def _sniff_delimiter(lines):
    """
    Try Python's built-in csv.Sniffer on a sample.

    Args:
        lines (list[str]): Sample lines (already decoded).

    Returns:
        str | None: The detected delimiter if sniffing succeeds; otherwise None.
    """
    # Join up to 50 lines to form the sample string Sniffer will inspect.
    sample = "\n".join(lines[:50])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        return dialect.delimiter
    except Exception:
        # Sniffer is known to be brittle on small/dirty samples—silently fallback.
        return None


def _heuristic_delimiter(lines):
    """
    Fallback delimiter detection based on count consistency per line.

    Heuristic:
      - For each candidate delimiter, count occurrences per line.
      - Keep only lines with count > 0 (line must contain the delimiter).
      - Require at least half of lines to contain the delimiter (min 2).
      - Compute the mode (most common count). If the proportion of lines that
        exhibit the modal count is >= 80%, accept that delimiter.

    Args:
        lines (list[str]): Sample lines.

    Returns:
        str | None: Best delimiter if one meets the consistency threshold; else None.
    """
    candidates = [",", "\t", ";", "|"]
    best = None
    best_score = 0.0

    for d in candidates:
        # Count how many times the delimiter appears in each line.
        counts = [ln.count(d) for ln in lines]
        # Consider only lines that actually contain the delimiter at least once.
        nonzero = [c for c in counts if c > 0]

        # Require that more than half of lines (and at least 2) contain the delimiter.
        if len(nonzero) < max(2, int(0.5 * len(lines))):
            continue

        # Find the modal count and its frequency.
        cnt = Counter(nonzero)
        pairs = cnt.most_common(1)
        if not pairs:
            continue

        mode, mode_freq = pairs[0]
        # Consistency ratio: lines with the modal count / total lines in the sample.
        consistency = mode_freq / len(lines)
        # Accept if consistent enough and better than any previous candidate.
        if mode >= 1 and consistency >= 0.80 and consistency > best_score:
            best = d
            best_score = consistency

    return best


def _lightweight_parse_check(lines, delimiter):
    """
    Parse a few lines with csv.reader and check structural stability.

    Heuristic:
      - Parse up to 5 lines with the given delimiter.
      - Count column widths per parsed row.
      - Require at least 2 non-empty rows.
      - Allow at most 1 row whose width deviates by >2 columns from the first row.

    Args:
        lines (list[str]): Sample lines (decoded).
        delimiter (str): Delimiter chosen by sniffing/heuristics.

    Returns:
        bool: True if parsing looks stable; False otherwise.
    """
    try:
        # csv.reader accepts any iterable of strings; feeding the first 10 lines is fine.
        reader = csv.reader(lines[:10], delimiter=delimiter)
        widths = []
        valid_rows = 0
        for row in reader:
            if not row:
                continue

            widths.append(len(row))
            valid_rows += 1

        # Need at least two meaningful rows to make a judgment.
        if valid_rows < 2:
            return False

        if widths:
            first = widths[0]
            # Count rows whose width deviates significantly (>2) from the first row.
            unstable = sum(1 for w in widths if abs(w - first) > 2)
            # Permit at most 1 unstable row among the parsed sample.
            return unstable <= 1
        return False
    except Exception:
        return False
