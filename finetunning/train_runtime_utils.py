import logging
import warnings


_TORCH_DTYPE_DEPRECATION = "`torch_dtype` is deprecated! Use `dtype` instead!"


class _DropTransformersTorchDtypeFilter(logging.Filter):
    def filter(self, record):
        return record.getMessage() != _TORCH_DTYPE_DEPRECATION


def configure_training_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r".*_check_is_size will be removed.*guard_size_oblivious.*",
        category=FutureWarning,
        module=r"bitsandbytes(\..*)?",
    )

    for logger_name in (
        "transformers.configuration_utils",
        "transformers.modeling_utils",
        "transformers.pipelines",
        "transformers.pipelines.base",
    ):
        logger = logging.getLogger(logger_name)
        if not any(
            isinstance(item, _DropTransformersTorchDtypeFilter)
            for item in logger.filters
        ):
            logger.addFilter(_DropTransformersTorchDtypeFilter())


def from_pretrained_with_dtype_fallback(model_class, model_name, kwargs):
    try:
        return model_class.from_pretrained(model_name, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc) or "dtype" not in kwargs:
            raise

        legacy_kwargs = dict(kwargs)
        legacy_kwargs["torch_dtype"] = legacy_kwargs.pop("dtype")
        return model_class.from_pretrained(model_name, **legacy_kwargs)
