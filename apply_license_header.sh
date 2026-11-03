#!/usr/bin/env sh
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: 2026 Forschungszentrum Jülich GmbH
# SPDX-License-Identifier: Apache-2.0
#
# Prepend an SPDX / copyright header to every Python source file,
# but only if the file does not already contain an SPDX identifier.
# The header is taken verbatim from the file given on the command line.

set -eu   # quit on error, treat unset vars as errors

# Helper / usage
print_usage() {
    printf 'Usage: %s HEADER_FILE [ROOT_DIR]\n' "$(basename "$0")" >&2
    printf '\n'
    printf '  HEADER_FILE   Path to a text file that contains the header to prepend.\n'
    printf '                The file must be readable.\n'
    printf '  ROOT_DIR      Optional directory to start the search from (default: .)\n' >&2
    exit 1
}

# Parse arguments
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    print_usage
fi

HEADER_FILE=$1            # first argument = header path
ROOT_DIR=${2:-.}          # optional second argument = base directory (default ".")

# Verify the header file is readable
if [ ! -r "$HEADER_FILE" ]; then
    printf 'Error: header file %s not found or not readable\n' "$HEADER_FILE" >&2
    exit 1
fi

# Add the header to a single file.
add_header() {
    file=$1

    # If the file already contains an SPDX‑License line, do nothing.
    if grep -q '^# SPDX-License-Identifier:' "$file"; then
        return
    fi

    # Grab the first line (may be empty if the file is empty)
    first_line=$(head -n 1 "$file" || true)

    # Work in the same directory – makes the final mv atomic even on
    # non‑POSIX‑compliant filesystems (no cross‑device rename).
    tmp="${file}.tmp.$$"

    case "$first_line" in
        \#\!*)   # she‑bang present → keep it, then insert the header
            printf '%s\n' "$first_line" > "$tmp"
            cat "$HEADER_FILE" >> "$tmp"
            tail -n +2 "$file" >> "$tmp"
            ;;
        *)      # no she‑bang → prepend header directly
            cat "$HEADER_FILE" > "$tmp"
            cat "$file" >> "$tmp"
            ;;
    esac

    # Replace atomically
    mv -f "$tmp" "$file"
    printf 'header added → %s\n' "$file"
}

# Walk the tree and process every relevant source file, skipping hidden dirs.
find "$ROOT_DIR" \
    -type f \( -name '*.py' -o -name '*.sh' -o -name '*.bash' \) -print |
while IFS= read -r file; do
    add_header "$file"
done
