#!/bin/bash
# ClipForge Auto-Loop — autonomous development tracker
#
# Run from the repo root:
#   bash scripts/auto_loop.sh status     # show current progress
#   bash scripts/auto_loop.sh next       # print next pending phase
#   bash scripts/auto_loop.sh done <phase> <commit-hash>  # mark phase done
#   bash scripts/auto_loop.sh set <phase> <key> <value>   # set progress field
#
# This script is used by the autonomous agent to track what's been done
# and determine what to work on next.

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATUS_FILE="$REPO_ROOT/phase-status.json"

PHASE_ORDER=(
  "phase-0:Foundation (Docker, CLI, errors, CI, releases)"
  "phase-1:Editor UX (shortcuts, undo/redo, captions, bulk ops)"
  "phase-2:Languages (FR, ES, PT, IT)"
  "phase-3:Feature parity (brand templates, URL progress)"
  "phase-4:Moats (watch folder, privacy mode, plugin system)"
)

cmd_status() {
  echo "=== ClipForge Phase Status ==="
  echo ""
  for entry in "${PHASE_ORDER[@]}"; do
    id="${entry%%:*}"
    label="${entry#*:}"
    status=$(grep -o "\"$id\":{[^}]*\"status\":\"[^\"]*\"" "$STATUS_FILE" 2>/dev/null | grep -o '"status":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
    commits=$(grep -o "\"$id\":{[^}]*\"commits\":\[[^]]*\]" "$STATUS_FILE" 2>/dev/null | grep -o '\[.*\]' || echo "[]")
    count=$(echo "$commits" | grep -o '","' | wc -l)
    if [ "$status" = "done" ]; then
      echo "  ✅ $id — $label ($((count+1)) commits)"
    elif [ "$status" = "in_progress" ]; then
      echo "  🔄 $id — $label (in progress)"
    else
      echo "  ⬜ $id — $label"
    fi
  done
  echo ""

  # Show git log since last tag
  LAST_TAG=$(git -C "$REPO_ROOT" describe --tags --abbrev=0 2>/dev/null || echo "none")
  AHEAD=$(git -C "$REPO_ROOT" rev-list --count "${LAST_TAG}..HEAD" 2>/dev/null || echo 0)
  echo "Branch: $(git -C "$REPO_ROOT" branch --show-current)"
  echo "Ahead of $LAST_TAG: $AHEAD commits"
}

cmd_next() {
  for entry in "${PHASE_ORDER[@]}"; do
    id="${entry%%:*}"
    label="${entry#*:}"
    status=$(grep -o "\"$id\":{[^}]*\"status\":\"[^\"]*\"" "$STATUS_FILE" 2>/dev/null | grep -o '"status":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
    if [ "$status" != "done" ]; then
      echo "$id:$label"
      exit 0
    fi
  done
  echo "ALL_DONE"
  exit 0
}

cmd_done() {
  local phase="$1"
  local commit="$2"
  if [ -z "$phase" ]; then
    echo "Usage: $0 done <phase> [commit-hash]"
    exit 1
  fi
  # Update status to done
  python3 -c "
import json
with open('$STATUS_FILE') as f:
    data = json.load(f)
if '$phase' in data.get('phases', {}):
    data['phases']['$phase']['status'] = 'done'
    if '$commit' and '$commit' not in data['phases']['$phase'].get('commits', []):
        data['phases']['$phase'].setdefault('commits', []).append('$commit')
    data['last_updated'] = '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
    # Auto-advance current_phase to next undone
    phases_order = ['$(echo "${PHASE_ORDER[@]}" | sed "s/ /','/g")']
    for pid, _ in [p.split(':') for p in phases_order]:
        if data['phases'].get(pid, {}).get('status') != 'done':
            data['current_phase'] = pid
            break
    with open('$STATUS_FILE', 'w') as f:
        json.dump(data, f, indent=2)
    print('Done')
" 2>/dev/null || {
    # Fallback: manual JSON edit via sed
    if grep -q "\"$phase\"" "$STATUS_FILE"; then
      sed -i "s/\"$phase\":{[^}]*\"status\":\"[^\"]*\"/\"$phase\":{\"label\":\"$(grep -o "\"$phase\":{[^}]*\"label\":\"[^\"]*\"" "$STATUS_FILE" | grep -o '"label":"[^"]*"' | cut -d'"' -f4)\",\"status\":\"done\"/" "$STATUS_FILE"
    fi
  }
}

cmd_set() {
  local phase="$1"
  local key="$2"
  local value="$3"
  if [ -z "$phase" ] || [ -z "$key" ] || [ -z "$value" ]; then
    echo "Usage: $0 set <phase> <key> <value>"
    exit 1
  fi
  python3 -c "
import json
with open('$STATUS_FILE') as f:
    data = json.load(f)
if '$phase' in data.get('phases', {}):
    data['phases']['$phase']['$key'] = '$value'
    with open('$STATUS_FILE', 'w') as f:
        json.dump(data, f, indent=2)
    print('Set $phase.$key = $value')
" 2>/dev/null || true
}

case "${1:-status}" in
  status) cmd_status ;;
  next)   cmd_next ;;
  done)   shift; cmd_done "$@" ;;
  set)    shift; cmd_set "$@" ;;
  *)
    echo "Usage: $0 {status|next|done <phase> [commit]|set <phase> <key> <value>}"
    exit 1
    ;;
esac
