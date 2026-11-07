#!/usr/bin/env bash
# Forward recorder, every day: PMU main + internet pool odds every 30 s from 20 min before to 20 min after each French race
# (src/fr_live_probe.py --pmu-only -> data/raw/pmu_live/<date>.jsonl). Stop it by killing this script's PID (see recorder.pid).
cd "$(dirname "$0")"
echo $$ > data/raw/pmu_live/recorder.pid
while true; do
  day=$(TZ=Europe/Paris date +%F)
  python3 -I src/fr_live_probe.py "$day" --pmu-only >> data/raw/pmu_live/recorder.log 2>&1
  next=$(TZ=Europe/Paris date -d "tomorrow 09:00" +%s); sleep $(( next - $(date +%s) ))
done
