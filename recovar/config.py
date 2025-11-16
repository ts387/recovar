import logging
import os
import platform

# Configure XLA memory allocation (CUDA-specific, ignored on Metal)
# Only set for non-Apple Silicon systems to avoid confusion
if not (platform.system() == 'Darwin' and platform.processor() == 'arm'):
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"

# import tensorflow as tf
# tf.config.experimental.set_visible_devices([], "GPU")
import jax

# Enable 64-bit precision for numerical accuracy
# Note: Metal has limited float64 support, some operations may fall back to CPU
jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)

try:
    devices = jax.devices()
    device_kinds = [d.device_kind for d in devices]
    logger.info(f"Devices found: {','.join(device_kinds)}")

    # Check for Metal backend and warn about limitations
    if any('metal' in str(d).lower() for d in devices):
        logger.info("Metal GPU backend detected (Apple Silicon).")
        logger.info("Note: Some float64/complex128 operations may fall back to CPU for Metal backend.")
        logger.info("This is expected behavior and should not affect correctness, only performance.")
except RuntimeError as e:
    logger.warning("---------------------------------------------------")
    logger.warning("---------------------------------------------------")
    logger.warning("No JAX devices found! Falling back to CPU-only mode.")
    logger.warning("---------------------------------------------------")
    logger.warning("---------------------------------------------------")

