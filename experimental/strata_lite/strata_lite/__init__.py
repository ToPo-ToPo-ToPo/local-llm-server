
# Keep experimental downloads on Python TLS until a patched hf-xet is verified.
import os as _os
import sys as _sys
_os.environ["HF_HUB_DISABLE_XET"] = "1"
_hf_constants = _sys.modules.get("huggingface_hub.constants")
if _hf_constants is not None:
    _hf_constants.HF_HUB_DISABLE_XET = True
