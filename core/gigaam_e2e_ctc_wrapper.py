r"""Lazy wrapper for GigaAM-v3 end-to-end CTC speech-to-text (CPU or CUDA).

Uses the official Transformers API::

    AutoModel.from_pretrained(..., revision="e2e_ctc", trust_remote_code=True)
    model.transcribe(path)

The v3 ``e2e_ctc`` head returns punctuated, capitalized text (BPE + ITN).
Windowing, ffmpeg fallback, cancel, result coercion, the device machinery, and
the opt-in pool modes (strict device, thread-safe lock-free path, window
batches) are inherited from :class:`~core.gigaam_ctc_wrapper.GigaAMCtcWrapper`.

Weights always load on CPU as fp32 under the transformers 5.x compatibility
patches. For ``--device auto|cuda`` the loaded model then moves to CUDA as a
whole (see :mod:`core.gigaam_device`); the remote-code encoder runs under its
own ``torch.autocast(float16)`` on CUDA while the mel front-end stays fp32.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any, ClassVar, cast

from core.gigaam_ctc_wrapper import GigaAMCtcWrapper, _raise_missing_extra
from core.stt_providers import (
    GIGAAM_V3_HF_REPO_ID,
    GIGAAM_V3_HF_REVISION,
    STT_PROVIDER_GIGAAM_E2E_CTC,
)

logger = logging.getLogger(__name__)

# * These patches target the transformers 5.x private API
# * (``PreTrainedModel._finalize_model_loading``) and were verified against
# * the transformers 5.x line pinned in ``[ml]`` (transformers>=5,<6).

# ! Serializes the global monkeypatch window of ``_load_model``: the external
# ! daemon runs one lazy pipeline per worker thread, so concurrent first loads
# ! would interleave the hydra/transformers patches — a thread could restore
# ! the originals mid-load or leak another thread's wrapper as the "original".
_LOAD_PATCH_LOCK = threading.Lock()

_HydraInstantiate = Callable[..., object]
_FinalizeLoading = Callable[..., object]
_FINALIZE_ATTR = "_finalize_model_loading"
_TIED_WEIGHTS_ATTR = "all_tied_weights_keys"
_POST_INIT_ATTR = "post_init"


def _get_finalize_loading(cls: object) -> _FinalizeLoading:
    """Return the current finalize-loading callable on *cls*."""
    return cast("_FinalizeLoading", getattr(cls, _FINALIZE_ATTR))


def _set_finalize_loading(cls: object, value: object) -> None:
    """Install *value* as the finalize-loading callable on *cls*."""
    setattr(cls, _FINALIZE_ATTR, value)


def _prepare_v3_tied_weights(model: object) -> None:
    """Ensure v3 remote-code models expose transformers-5 tied-weight state."""
    if not hasattr(model, _TIED_WEIGHTS_ATTR) and hasattr(model, _POST_INIT_ATTR):
        getattr(model, _POST_INIT_ATTR)()
    if not hasattr(model, _TIED_WEIGHTS_ATTR):
        setattr(model, _TIED_WEIGHTS_ATTR, {})


def _patch_hydra_instantiate_on_cpu() -> _HydraInstantiate:
    """Force hydra.instantiate onto CPU.

    GigaAM-v3 remote-code has no transformers-5 meta-device guard.
    Multilingual modeling wraps ``instantiate`` in ``torch.device("cpu")``;
    v3 does not, so MelSpectrogram filterbanks mix cpu/meta and crash.
    ``low_cpu_mem_usage`` is ignored in transformers 5.x, so patch hydra.

    Returns:
        The original ``hydra.utils.instantiate`` callable.

    """
    import hydra.utils as hydra_utils  # noqa: PLC0415
    import torch  # noqa: PLC0415

    original = cast("_HydraInstantiate", hydra_utils.instantiate)

    def _cpu_instantiate(*args: object, **kwargs: object) -> object:
        with torch.device("cpu"):
            return original(*args, **kwargs)

    hydra_utils.instantiate = _cpu_instantiate
    return original


def _patch_v3_finalize_loading() -> _FinalizeLoading:
    """Fill missing transformers-5 attributes on v3 GigaAMModel before finalize.

    v3 remote-code never calls ``post_init()`` (multilingual does), so
    ``all_tied_weights_keys`` is absent and load crashes.

    Returns:
        The original ``PreTrainedModel._finalize_model_loading`` callable.

    """
    from transformers.modeling_utils import PreTrainedModel  # noqa: PLC0415

    original = _get_finalize_loading(PreTrainedModel)

    def _finalize(model: object, load_config: object, loading_info: object) -> object:
        _prepare_v3_tied_weights(model)
        return original(model, load_config, loading_info)

    _set_finalize_loading(PreTrainedModel, staticmethod(_finalize))
    return original


class GigaAME2ECtcWrapper(GigaAMCtcWrapper):
    """Load GigaAM-v3 e2e CTC on CPU, optionally move it to CUDA, and transcribe."""

    # * CUDA-capable head: ``device`` is ``auto``/``cpu``/``cuda``, resolved at load.
    PROVIDER: ClassVar[str] = STT_PROVIDER_GIGAAM_E2E_CTC

    def _load_model(self) -> None:
        """Load the pinned GigaAM-v3 e2e_ctc revision when not already loaded.

        Applies transformers 5.x compatibility patches around
        ``from_pretrained`` and always restores the original globals, including
        when load raises. Weights load on CPU as fp32; the device request is
        applied after the patches are restored, so a CUDA move never extends
        the global patch window.
        """
        if self._model is not None:
            return
        auto_model_cls, torch_mod = self._ensure_deps()
        # * Strict device check and torch threads apply before any weight load.
        self._prepare_load(torch_mod)
        kwargs: dict[str, Any] = {
            "revision": GIGAAM_V3_HF_REVISION,
            "trust_remote_code": True,
        }
        if self.cache_dir is not None:
            kwargs["cache_dir"] = str(self.cache_dir)
        logger.info(
            "Loading GigaAM-v3 e2e CTC: repo=%s revision=%s load_device=cpu "
            "requested_device=%s",
            GIGAAM_V3_HF_REPO_ID,
            GIGAAM_V3_HF_REVISION,
            self.device,
        )
        try:
            import hydra.utils as hydra_utils  # noqa: PLC0415
            from transformers.modeling_utils import (  # noqa: PLC0415
                PreTrainedModel,
            )
        except ImportError as exc:
            _raise_missing_extra(exc)

        original_instantiate: _HydraInstantiate | None = None
        original_finalize: _FinalizeLoading | None = None
        loaded: Any = None
        with _LOAD_PATCH_LOCK:
            try:
                # * Patch hydra.utils.instantiate before remote code imports it.
                original_instantiate = _patch_hydra_instantiate_on_cpu()
                original_finalize = _patch_v3_finalize_loading()
                # * Remote-code modeling may ImportError on sentencepiece/pyannote
                # * even after torch/transformers import; map that to the install hint.
                try:
                    model = auto_model_cls.from_pretrained(
                        GIGAAM_V3_HF_REPO_ID, **kwargs
                    )
                except ImportError as exc:
                    _raise_missing_extra(exc)
                # * v3 bins on disk look fp16; fp32 is required for Mel/STFT parity.
                # ! Never .half() the model: the n_fft=320 mel front-end must stay
                # ! fp32 (cuFFT fp16 needs power-of-two sizes); the encoder uses the
                # ! remote code's own fp16 autocast on CUDA.
                model = model.float()
                if hasattr(model, "eval"):
                    model.eval()
                loaded = model
            finally:
                if original_instantiate is not None:
                    hydra_utils.instantiate = original_instantiate
                if original_finalize is not None:
                    _set_finalize_loading(PreTrainedModel, original_finalize)
        self._install_loaded_model(loaded, torch_mod)
