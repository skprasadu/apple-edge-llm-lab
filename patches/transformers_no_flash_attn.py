# custom_patches/transformers_no_flash_attn.py
from contextlib import contextmanager

@contextmanager
def no_flash_attn_imports():
    """
    Some repos hard-require flash_attn at import time.
    On Apple/MPS this is often irrelevant but can crash loading.

    This patches transformers.dynamic_module_utils.get_imports to remove
    'flash_attn' so the model can load and use SDPA/eager instead.
    """
    try:
        import transformers.dynamic_module_utils as dmu
    except Exception as e:
        raise RuntimeError(
            "transformers.dynamic_module_utils not found. "
            "Confirm you installed 'transformers' and it's importable."
        ) from e

    original = dmu.get_imports

    def patched_get_imports(filename):
        imports = original(filename)
        # Be defensive: some repos reference different names.
        for bad in ("flash_attn", "flash_attn_2"):
            if bad in imports:
                imports.remove(bad)
        return imports

    dmu.get_imports = patched_get_imports
    try:
        yield
    finally:
        dmu.get_imports = original