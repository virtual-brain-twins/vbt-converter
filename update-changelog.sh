#!/usr/bin/env bash
#=================================================================
# update-changelog.sh
#
#   Reads .release-summary.txt and injects it immediately
#   after the version header in CHANGELOG.md.
#
#   Requirements:
#       * node  (to read package.json version)
#       * GNU awk / mawk (both support -v with new‑lines)
#=================================================================
set -euo pipefail

# Helper to print an error and exit
error() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

# Make sure we have everything we need
[[ -f .release-summary.txt ]]   || error "No .release-summary.txt – nothing to do."
[[ -f package.json ]]           || error "package.json not found."
[[ -f CHANGELOG.md ]]           || error "CHANGELOG.md not found."

# Pull the version number from package.json
if ! VERSION=$(node -p "require('./package.json').version" 2>/dev/null); then
    error "Unable to read version from package.json (node missing or invalid JSON?)."
fi

# Read the summary *exactly as‑is* and prepare it for the -v
#      argument.
#   • Escape backslashes   → '\\'   (so awk sees a literal \)
#   • Escape double quotes → '\"'   (so the shell doesn’t close the string)
#   • Do NOT escape new‑lines – they should stay real new‑lines
ESCAPED_SUM=$(sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' .release-summary.txt)

# Insert the summary after the matching version header.
#   • The header line can be any of the common forms, for example:
#        ## 1.1.0
#        ## 1.1.0 (2023‑09‑21)
#        ## 1.1.0 – 2023‑09‑21
#        ## 1.1.0 (https://…)(2026‑09‑17)
#   • We simply look for a line that starts with "##", optional blanks,
#     and the exact version number.  Anything that follows the version
#     (URL, date, dash, …) is accepted.
awk -v ver="${VERSION}" -v txt="${ESCAPED_SUM}" '
    # Header line – accept any characters after the version number.
    $0 ~ "^##[[:space:]]*" ver {
        print               # keep the original header line
        print txt           # insert the whole multi‑line summary
        inserted = 1       # flag for later sanity check
        next                # skip further processing for this line
    }
    { print }               # everything else is copied unchanged
    END {
        if (!inserted) {
            printf "ERROR: Could not find a version header for %s in CHANGELOG.md\n", ver > "/dev/stderr"
            exit 1
        }
    }
' CHANGELOG.md > CHANGELOG.tmp && mv CHANGELOG.tmp CHANGELOG.md

# Clean up
rm -f .release-summary.txt

# Success message
printf 'Summary successfully inserted for version %s.\n' "${VERSION}"