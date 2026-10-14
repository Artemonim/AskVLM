"""STT concurrency benchmark helpers for ``tools/benchmark_stt_concurrency.py``.

The package measures how resident STT engines (faster-whisper ``small`` and
GigaAM-v3 ``e2e_ctc``) scale under three concurrency modes:

* ``P`` - N worker processes, each holding its own model;
* ``S`` - one process holding one model served by N concurrent Python threads;
* ``B`` - one process holding one model that batches N items per forward pass.

Every configuration runs in freshly spawned child processes so that memory,
thread pools, and CUDA state never leak from one configuration into the next.
"""
