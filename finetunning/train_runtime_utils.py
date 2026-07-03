import logging
import math
import time
import warnings

from transformers import TrainerCallback


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


def _interval_value(value):
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return 0
    return numeric if numeric > 0 else 0


def _is_steps_strategy(value):
    return str(value).lower().endswith("steps")


def _format_duration(seconds):
    if seconds is None or seconds <= 0:
        return "unknown"

    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _next_interval_step(global_step, interval, max_steps):
    if interval <= 0 or global_step >= max_steps:
        return None

    next_step = ((global_step // interval) + 1) * interval
    return next_step if next_step <= max_steps else None


class GlobalProgressCallback(TrainerCallback):
    def __init__(self, train_size, eval_size=None):
        self.train_size = int(train_size)
        self.eval_size = int(eval_size) if eval_size is not None else None
        self.started_at = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.started_at = time.monotonic()
        max_steps = max(1, int(state.max_steps or args.max_steps or 0))
        per_device_batch = int(args.per_device_train_batch_size)
        world_size = max(1, int(getattr(args, "world_size", 1)))
        train_batch = per_device_batch * world_size
        effective_batch = train_batch * int(args.gradient_accumulation_steps)
        examples_per_epoch = self.train_size
        estimated_steps_per_epoch = max(1, math.ceil(examples_per_epoch / effective_batch))
        requested_epochs = float(args.num_train_epochs)

        if args.max_steps and args.max_steps > 0:
            planned_epochs = max_steps / estimated_steps_per_epoch
        else:
            planned_epochs = requested_epochs

        eval_steps = (
            _interval_value(args.eval_steps)
            if _is_steps_strategy(getattr(args, "eval_strategy", None))
            or _is_steps_strategy(getattr(args, "evaluation_strategy", None))
            else 0
        )
        save_steps = (
            _interval_value(args.save_steps)
            if _is_steps_strategy(getattr(args, "save_strategy", None))
            else 0
        )
        logging_steps = _interval_value(args.logging_steps)
        eval_count = max_steps // eval_steps if eval_steps else 0
        save_count = max_steps // save_steps if save_steps else 0

        print("\n=== Global training plan ===")
        print(f"Train examples: {examples_per_epoch:,}")
        if self.eval_size is not None:
            print(f"Validation examples: {self.eval_size:,}")
        print(
            "Effective batch: "
            f"{effective_batch:,} "
            f"({per_device_batch} per-device x {world_size} GPU x "
            f"{args.gradient_accumulation_steps} grad_accum)"
        )
        print(f"Steps per epoch: ~{estimated_steps_per_epoch:,}")
        print(f"Planned epochs: ~{planned_epochs:.2f}")
        print(f"Total optimizer steps: {max_steps:,}")
        print(
            "Intervals: "
            f"log every {logging_steps or 'disabled'} steps, "
            f"eval every {eval_steps or 'disabled'} steps (~{eval_count} evals), "
            f"save every {save_steps or 'disabled'} steps (~{save_count} saves)"
        )
        print("============================\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        max_steps = int(state.max_steps or args.max_steps or 0)
        global_step = int(state.global_step)
        if max_steps <= 0 or global_step <= 0:
            return

        pct = min(100.0, 100.0 * global_step / max_steps)
        elapsed = time.monotonic() - self.started_at if self.started_at else None
        eta = None
        if elapsed and global_step:
            eta = elapsed * (max_steps - global_step) / global_step

        eval_steps = (
            _interval_value(args.eval_steps)
            if _is_steps_strategy(getattr(args, "eval_strategy", None))
            or _is_steps_strategy(getattr(args, "evaluation_strategy", None))
            else 0
        )
        save_steps = (
            _interval_value(args.save_steps)
            if _is_steps_strategy(getattr(args, "save_strategy", None))
            else 0
        )
        next_eval = _next_interval_step(global_step, eval_steps, max_steps)
        next_save = _next_interval_step(global_step, save_steps, max_steps)
        epoch = state.epoch if state.epoch is not None else 0

        print(
            "[global] "
            f"step {global_step:,}/{max_steps:,} ({pct:.1f}%) | "
            f"epoch {epoch:.2f} | "
            f"elapsed {_format_duration(elapsed)} | "
            f"eta {_format_duration(eta)} | "
            f"next_eval {next_eval or '-'} | "
            f"next_save {next_save or '-'}"
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        max_steps = int(state.max_steps or args.max_steps or 0)
        print(f"[eval] completed at step {state.global_step:,}/{max_steps:,}")

    def on_save(self, args, state, control, **kwargs):
        max_steps = int(state.max_steps or args.max_steps or 0)
        print(f"[save] checkpoint at step {state.global_step:,}/{max_steps:,}")
