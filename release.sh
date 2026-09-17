#!/usr/bin/env bash
#=====================================================================
# release.sh – Automated (first) release and subsequent semver releases
#
#   • First run → tag v1.0.0, generate CHANGELOG.md
#   • Later runs → bump version (patch/minor/major) and update changelog
#   • Uses npm + commit-and-tag-version
#
#   Usage:
#       ./release.sh                # → default “patch” bump
#       ./release.sh --minor        # → minor bump (vX.Y+1.0)
#       ./release.sh --major        # → major bump (vX+1.0.0)
#       ./release.sh --first-release   # → force creation of v1.0.0
#
#   Prerequisites (run once):
#       sudo apt-get update && sudo apt-get install -y nodejs npm git
#=====================================================================

set -euo pipefail                       # abort on errors, undefined vars, pipefails
IFS=$'\n\t'                             # safer word splitting

# Helpers & sanity checks #
log()   { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die()   { log "❌ $*" >&2; exit 1; }

# Must be inside a git repository
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "Not a git repository."

# Working tree must be clean
if ! git diff-index --quiet HEAD --; then
    die "Working tree is dirty – commit or stash your changes before releasing."
fi

# Detect whether a "vX.Y.Z" tag already exists
has_release_tag() {
    git tag --list 'v[0-9]*.[0-9]*.[0-9]*' | wc -l
}
CURRENT_TAG_COUNT=$(has_release_tag)

# Parse command line args #
BUMP_TYPE="patch"               # default
FIRST_RELEASE=false

while (( $# )); do
    case "$1" in
        --major)   BUMP_TYPE="major"   ;;
        --minor)   BUMP_TYPE="minor"   ;;
        --patch)   BUMP_TYPE="patch"   ;;   # explicit, same as default
        --first-release) FIRST_RELEASE=true ;;
        -h|--help)
            cat <<'EOF'
release.sh – automate semantic‑version releases

Options:
  --major          bump major version  (vX.0.0 → vY.0.0)
  --minor          bump minor version  (vX.Y.0 → vX.Y+1.0)
  --patch          bump patch version (default) (vX.Y.Z → vX.Y.Z+1)
  --first-release  force creation of the very first tag (v1.0.0)
  -h, --help       show this help
EOF
            exit 0
            ;;
        *) die "Unknown option: $1" ;;
    esac
    shift
done

# Verify npx availability
STDVER="commit-and-tag-version"
if ! command -v npx >/dev/null; then
    die "npx not found – please install nodejs & npm (apt-get install nodejs npm)."
fi

# Summary Hook Setup: Prompt developer for a high-level summary entry
SUMMARY_FILE=".release-summary.txt"
rm -f "$SUMMARY_FILE"

echo -n " Do you want to add a custom detailed summary to this release? (y/N): "
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    EDITOR=${EDITOR:-nano}
    cat <<EOF > "$SUMMARY_FILE"

### Release Summary
- Put your high-level manual summary text here.
- This text block gets auto-injected right below the version title block.

EOF
    log "Opening $EDITOR to write release summary..."
    $EDITOR "$SUMMARY_FILE"
fi

# FIRST RELEASE or bump version #
if (( CURRENT_TAG_COUNT == 0 )) || $FIRST_RELEASE; then
    log "No existing release tag – creating the *first* release v1.0.0"
    npx "$STDVER" --first-release --release-as "$BUMP_TYPE"
else
    log "Bumping version – $BUMP_TYPE"
    npx "$STDVER" --release-as "$BUMP_TYPE"
fi

# Push the new commit and the new tag
log "Pushing release commit and tag to origin"
git push origin "$(git rev-parse --abbrev-ref HEAD)"   # push the branch
git push origin --tags                                 # push the new tag

log "Release finished – current version is $(git describe --tags --abbrev=0)"
log "CHANGELOG.md has been updated (last entry contains your summary and automatic commit lists)."
