#!/bin/sh
# download_child_artifacts.sh - Script to fetch GitLab child pipeline artifacts

# LOGGING SETUP
if [ -n "$VBT_JOB_LOGS" ]; then
    # Ensure the directory exists
    mkdir -p "$VBT_JOB_LOGS"

    # Build the full log‑file path  (e.g. /path/to/logs/download_child_artifacts.log)
    SCRIPT_NAME="$(basename "$0" .sh)"
    LOG_FILE="${VBT_JOB_LOGS}/${SCRIPT_NAME}.log"

    # Create / truncate the log file so we start fresh for this run
    : > "$LOG_FILE"
else
    echo "ERROR: Environment variable VBT_JOB_LOGS is not defined." >&2
    exit 1
fi

log_message() {
    _timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$_timestamp] $1" >> "$LOG_FILE"
    echo "$1"
}

# PARAMETER VALIDATION
TRIGGER_JOB_NAME="$1"
TARGET_CHILD_JOB_NAME="$2"

if [ -z "$TRIGGER_JOB_NAME" ] || [ -z "$TARGET_CHILD_JOB_NAME" ]; then
    log_message "ERROR: Missing required parameters."
    log_message "Usage: $0 <TRIGGER_JOB_NAME> <TARGET_CHILD_JOB_NAME>"
    exit 1
fi

log_message "=== Starting Artifact Retrieval Process ==="
log_message "Trigger Job Name: $TRIGGER_JOB_NAME"
log_message "Target Child Job Name: $TARGET_CHILD_JOB_NAME"

# VERIFY MANDATORY GITLAB CI ENVIRONMENT VARIABLES
if [ -z "$CI_JOB_TOKEN" ] || [ -z "$CI_API_V4_URL" ] || [ -z "$CI_PROJECT_ID" ] || [ -z "$CI_PIPELINE_ID" ]; then
    log_message "ERROR: Missing mandatory GitLab CI environment variables."
    exit 1
fi

log_message "Fetching bridges for Parent Pipeline ID: $CI_PIPELINE_ID"

# GET CHILD PIPELINE ID FROM BRIDGE JOBS
_bridge_url="${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/pipelines/${CI_PIPELINE_ID}/bridges"
_bridge_response=$(curl --silent --fail --header "JOB-TOKEN: $CI_JOB_TOKEN" "$_bridge_url")

if [ $? -ne 0 ] || [ -z "$_bridge_response" ]; then
    log_message "ERROR: Failed to communicate with GitLab API or no bridge jobs found."
    exit 1
fi

# Extract downstream pipeline ID for the trigger job
CHILD_PIPELINE_ID=$(echo "$_bridge_response" | jq -r ".[] | select(.name==\"$TRIGGER_JOB_NAME\") | .downstream_pipeline.id // empty")

if [ -z "$CHILD_PIPELINE_ID" ] || [ "$CHILD_PIPELINE_ID" = "null" ]; then
    log_message "INFO: Trigger job '$TRIGGER_JOB_NAME' did not run or skipped. Skipping artifact download."
    log_message "=== Process Finished Comfortably (No Action Required) ==="
    exit 0
fi

log_message "Success: Found Child Pipeline ID: $CHILD_PIPELINE_ID"

# LIST JOBS IN CHILD PIPELINE
_jobs_url="${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/pipelines/${CHILD_PIPELINE_ID}/jobs"
_jobs_response=$(curl --silent --fail --header "JOB-TOKEN: $CI_JOB_TOKEN" "$_jobs_url")

if [ $? -ne 0 ] || [ -z "$_jobs_response" ]; then
    log_message "ERROR: Failed to retrieve jobs for child pipeline $CHILD_PIPELINE_ID."
    exit 1
fi

# Find the job ID of the target child job
CHILD_JOB_ID=$(echo "$_jobs_response" | jq -r ".[] | select(.name==\"$TARGET_CHILD_JOB_NAME\") | .id // empty")

if [ -z "$CHILD_JOB_ID" ] || [ "$CHILD_JOB_ID" = "null" ]; then
    log_message "WARNING: Target job '$TARGET_CHILD_JOB_NAME' was not executed in child pipeline $CHILD_PIPELINE_ID."
    log_message "=== Process Finished ==="
    exit 0
fi

log_message "Success: Found Target Child Job ID: $CHILD_JOB_ID"

# DOWNLOAD AND EXTRACT ARTIFACTS
_artifact_url="${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/jobs/${CHILD_JOB_ID}/artifacts"
_output_zip="child_${CHILD_JOB_ID}_artifacts.zip"

log_message "Downloading artifacts archive to $_output_zip..."
curl --silent --show-error --fail --location --header "JOB-TOKEN: $CI_JOB_TOKEN" --output "$_output_zip" "$_artifact_url"

if [ $? -ne 0 ]; then
    log_message "ERROR: Failed downloading artifacts archive from Job ID $CHILD_JOB_ID."
    exit 1
fi

log_message "Extracting artifacts archive..."
if unzip -o "$_output_zip"; then
    log_message "SUCCESS: Artifacts successfully fetched and unpacked from child job '$TARGET_CHILD_JOB_NAME'."
    rm -f "$_output_zip"
else
    log_message "ERROR: Failed extracting archive $_output_zip. Ensure 'unzip' utility is available."
    exit 1
fi

log_message "=== Process Completed Successfully ==="