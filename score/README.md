# Legacy Score Scripts

The files in this directory are the existing runnable pipeline entrypoints.

They remain here for compatibility with the current workflow and result filenames.

## Naming note

The filenames are legacy stage names such as:

- `0_score.py`
- `1_score.py`
- `1_converge.py`
- `4_0_mutual_score.py`

This naming is preserved because downstream outputs and habits already depend on it.

## Normalized project structure

New shared structure is now defined under:

- `evaluation/datasets.py`
- `evaluation/pipeline.py`
- `evaluation/io.py`
- `evaluation/parsing.py`

When adding new functionality, prefer putting shared logic there first, then letting legacy scripts consume it gradually.

## Essay expansion

Do not put future essay-specific shared code directly into these legacy classroom scripts unless it is only a temporary bridge.

Prefer:

1. adding reusable code under `evaluation/`
2. defining the essay dataset shape in `evaluation/datasets.py`
3. creating essay-specific entrypoints only after the shared layer is in place
