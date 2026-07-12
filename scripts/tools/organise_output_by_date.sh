#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
#  organise_output_by_date.sh
#
#  Scans a directory for files whose names contain a YYYY-MM-DD date suffix,
#  creates a sub-folder for each date, and moves the matching files into it.
#
#  Usage (run from Git Bash):
#    bash scripts/tools/organise_output_by_date.sh
#
#  Optional: pass a target directory as the first argument:
#    bash scripts/tools/organise_output_by_date.sh "output/us_local/2023-12-07/output"
#
#  Default target directory if no argument given:
TARGET_DIR="${1:-output/us_local/2023-12-07/output}"
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# Resolve to absolute path so we can cd safely
ABS_TARGET="$(cd "$TARGET_DIR" && pwd)"

echo "============================================================"
echo "  organise_output_by_date.sh"
echo "  Target : $ABS_TARGET"
echo "============================================================"

# Regex: match a YYYY-MM-DD date anywhere in the filename (before the extension)
DATE_PATTERN='([0-9]{4}-[0-9]{2}-[0-9]{2})'

moved=0
skipped=0
dates_created=()

for filepath in "$ABS_TARGET"/*; do
    # Only process regular files (skip sub-directories)
    [[ -f "$filepath" ]] || continue

    filename="$(basename "$filepath")"

    # Extract date from filename
    if [[ "$filename" =~ $DATE_PATTERN ]]; then
        date="${BASH_REMATCH[1]}"
        dest_dir="$ABS_TARGET/$date"

        # Create the date folder if it doesn't exist
        if [[ ! -d "$dest_dir" ]]; then
            mkdir -p "$dest_dir"
            dates_created+=("$date")
            echo "  Created folder: $date/"
        fi

        # Move the file
        mv "$filepath" "$dest_dir/$filename"
        echo "  Moved  : $filename  →  $date/"
        moved=$((moved + 1))
    else
        echo "  Skipped: $filename  (no date found)"
        skipped=$((skipped + 1))
    fi
done

echo ""
echo "============================================================"
echo "  Done."
echo "  Folders created : ${#dates_created[@]}"
echo "  Files moved     : $moved"
echo "  Files skipped   : $skipped"
echo "============================================================"
