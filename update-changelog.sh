#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: Apache-2.0

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
#       ## 1.1.0 …
#       ## [1.1.0] …
#       ## [1.1.0](…) …
#   • We simply look for a line that starts with "##", optional blanks,
#     and the exact version number.  Anything that follows the version
#     (URL, date, dash, …) is accepted.
printf 'Inserting summary for version %s into CHANGELOG.md...\n' "${VERSION}"
awk -v ver="${VERSION}" -v txt="${ESCAPED_SUM}" '
    #   ^##            → line starts with “##”
    #   [[:space:]]*   → optional whitespace
    #   \[?            → optional opening bracket
    #   ver            → the version number
    #   \]?            → optional closing bracket
    #   (\\([^)]*\\))? → optional “( … )” part (the markdown link)
    $0 ~ "^##[[:space:]]*\\[?" ver "\\]?(\\([^)]*\\))?" {
        print               # keep the original heading
        print txt           # inject the multi‑line summary
        inserted = 1
        next
    }
    { print }
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