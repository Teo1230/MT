import csv
from collections import Counter, defaultdict

# =======================
# CONFIG
# =======================

CSV_FILES = [
    "text2emoji_ro_test.csv",
    "text2emoji_ro_train.csv",
    "text2emoji_ro_valid.csv"
]

EMOJI_COL = "emoji"
TEXT_COL = "ro"

# =======================
#
# =======================

def split_emojis(s: str):
    emojis = []
    for ch in s:
        if ch.strip() == "":
            continue
        if ch in {"\u200d", "\ufe0f"}:  # ZWJ + variation selector
            continue
        emojis.append(ch)
    return emojis

# =======================
# 1. CITIM TOATE CSV-URILE, COLECTĂM FRECVENȚELE
# =======================

all_rows = []       # lista cu (filename, row_dict)
freq = Counter()    # emoji -> count

for file in CSV_FILES:
    with open(file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emoji_str = row.get(EMOJI_COL, "")
            emojis = split_emojis(emoji_str)
            for e in emojis:
                freq[e] += 1
            all_rows.append((file, row))

# =======================
# =======================

rare_emojis = {e for e, c in freq.items() if c == 1 or c == 2 or c == 3 or c == 4 or c == 5 or c == 6 or c == 7 }

print( rare_emojis)

# =======================
# =======================

cleaned = {file: [] for file in CSV_FILES}

for file, row in all_rows:
    emojis = split_emojis(row[EMOJI_COL])
    if any(e in rare_emojis for e in emojis):
        continue
    cleaned[file].append(row)

# =======================
# PRINT CSV
# =======================

for file in CSV_FILES:
    if cleaned[file]:
        fieldnames = cleaned[file][0].keys()
    else:
        with open(file, "r", encoding="utf-8") as f:
            fieldnames = csv.DictReader(f).fieldnames

    out_file = file.replace(".csv", "_clean.csv")

    with open(out_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned[file])

