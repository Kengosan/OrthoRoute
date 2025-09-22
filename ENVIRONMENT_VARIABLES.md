# OrthoRoute Environment Variables

Complete reference for environment variables that control OrthoRoute behavior.

## Core Routing Control

### ORTHO_BATCH
- **Purpose**: Override batch size for routing
- **Default**: 32
- **Values**: 1-64
- **Usage**: `set ORTHO_BATCH=2` for small test runs

### ORTHO_CPU_ONLY
- **Purpose**: Force CPU-only mode, disable GPU acceleration
- **Default**: 0 (auto-detect)
- **Values**: 0, 1
- **Usage**: `set ORTHO_CPU_ONLY=1`

### ORTHO_NET_LIMIT
- **Purpose**: Limit number of nets to route (testing)
- **Default**: 0 (no limit)
- **Values**: Any positive integer
- **Usage**: `set ORTHO_NET_LIMIT=10`

## Stop Conditions & Convergence

### ORTHO_DISABLE_EARLY_STOP
- **Purpose**: Force routing to run to max iterations
- **Default**: 0 (early stop enabled)
- **Values**: 0, 1, true, yes
- **Usage**: `set ORTHO_DISABLE_EARLY_STOP=1`

### ORTHO_CAPACITY_END
- **Purpose**: Emit detailed capacity analysis on routing failure
- **Default**: 0
- **Values**: 0, 1, true, yes
- **Usage**: `set ORTHO_CAPACITY_END=1`

## Debugging & Visualization

### ORTHO_SHOW_PORTALS
- **Purpose**: Show pad stubs in GUI by default
- **Default**: 1 (enabled)
- **Values**: 0, 1, true, yes
- **Usage**: `set ORTHO_SHOW_PORTALS=0`

## Determinism & Repro

### ORTHO_SEED
- **Purpose**: Seed for random number generator (deterministic routing)
- **Default**: 42
- **Values**: Any integer
- **Usage**: `set ORTHO_SEED=12345`

### ORTHO_DUMP_REPRO
- **Purpose**: Dump repro bundle JSON file on capacity failure
- **Default**: 0
- **Values**: 0, 1, true, yes
- **Usage**: `set ORTHO_DUMP_REPRO=1`

## Performance Tuning

### ORTHO_MODE
- **Purpose**: Routing mode selection
- **Default**: near_far
- **Values**: near_far, full
- **Usage**: `set ORTHO_MODE=full`

### ORTHO_PER_NET_BUDGET
- **Purpose**: Time budget per net (seconds)
- **Default**: varies by mode
- **Values**: Float > 0
- **Usage**: `set ORTHO_PER_NET_BUDGET=0.1`

## GPU Configuration

### ORTHO_GPU
- **Purpose**: Enable GPU acceleration
- **Default**: 0 (CPU only)
- **Values**: 0, 1
- **Usage**: `set ORTHO_GPU=1`

## Quick Test Configurations

### Minimal Test Run
```batch
set ORTHO_BATCH=1
set ORTHO_CPU_ONLY=1
set ORTHO_NET_LIMIT=5
python main.py --test-manhattan
```

### Full Debug Run
```batch
set ORTHO_CAPACITY_END=1
set ORTHO_SHOW_PORTALS=1
set ORTHO_DUMP_REPRO=1
set ORTHO_DISABLE_EARLY_STOP=1
python main.py --test-manhattan
```

### Deterministic Comparison
```batch
set ORTHO_SEED=12345
set ORTHO_CPU_ONLY=1
python main.py --test-manhattan
```

## Bisection Debugging

For systematic debugging, use these environment variables to isolate issues:

1. **Start minimal**: `ORTHO_BATCH=1 ORTHO_CPU_ONLY=1 ORTHO_NET_LIMIT=1`
2. **Add capacity analysis**: `ORTHO_CAPACITY_END=1`
3. **Force completion**: `ORTHO_DISABLE_EARLY_STOP=1`
4. **Enable repro**: `ORTHO_DUMP_REPRO=1 ORTHO_SEED=42`
5. **Full debugging**: Add `ORTHO_SHOW_PORTALS=1`

This systematic approach allows quick bisection of routing problems.