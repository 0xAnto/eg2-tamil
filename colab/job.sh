#!/bin/bash
# Usage on the VM: nohup bash colab/job.sh NAME CMD... &
# Runs CMD in /content/eg2-tamil. Writes /content/logs/NAME.log. The last log line is "JOB EXIT <code>".
# ponytail: no retry or stall watchdog here; the local poller stops the VM on failure or stall.
name=$1; shift
mkdir -p /content/logs
exec > "/content/logs/$name.log" 2>&1
trap 'echo "JOB EXIT $?"' EXIT
set -euo pipefail
cd /content/eg2-tamil
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
"$@"
