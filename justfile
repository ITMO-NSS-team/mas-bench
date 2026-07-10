# List discovered benchmarks and systems (only systems with installed deps appear).
available:
    uv run --no-sync python -c "from benchlib.benchmarks import discover; from benchlib.adapters import discover_adapters; print('benchmarks:', list(discover())); print('systems:', discover_adapters())"

# Run experiments (cross-OS); pass any CLI flag. See all with: just run --help
run *ARGS:
    uv run --no-sync python -m benchlib.cli {{ARGS}}
