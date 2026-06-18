#!/usr/bin/env sh

format_duration() {
  total_seconds=$1
  minutes=$((total_seconds / 60))
  seconds=$((total_seconds % 60))

  if [ "$minutes" -gt 0 ]; then
    printf "%d min %02d sec" "$minutes" "$seconds"
  else
    printf "%d sec" "$seconds"
  fi
}

start_time=$(date +%s)
"$@"
status=$?
end_time=$(date +%s)
elapsed=$((end_time - start_time))
duration=$(format_duration "$elapsed")

if [ "$status" -eq 0 ]; then
  printf "    Finished in: %s\n" "$duration"
else
  printf "    Failed after: %s\n" "$duration" >&2
fi

exit "$status"
