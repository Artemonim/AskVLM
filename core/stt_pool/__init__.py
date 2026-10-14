"""STT supervisor with a per-configuration worker pool.

The modules in this package that the thin client and the supervisor import at
module level stay free of ML packages (torch, transformers, hydra,
faster_whisper); only the engine module inside worker processes loads them.
"""
