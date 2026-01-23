import time
from typing import Optional, Callable
import logging


class _NullTimer:
    """A no-op timer returned when verbosity is falsy to avoid overhead."""
    def __init__(self, *args, **kwargs):
        self.last = 0.0
        self.total = 0.0
        self.count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def elapsed(self) -> float:
        return 0.0

    def __call__(self, fn: Callable):
        # return raw function to avoid wrapper overhead
        return fn


class Runtimer:
    """Context manager / decorator for simple runtime measurements.

    If `verbosity` is False, the factory returns a `_NullTimer` so that the
    `with`-block (or decorator) has near-zero overhead.

    Usage:
      timer = Runtimer(verbosity=True, name="step")
      with timer:
          ...

      @Runtimer(True, name="func")
      def f(...):
          ...
    """
    def __new__(cls, verbosity: bool = False, name: Optional[str] = None, logger: Optional[logging.Logger] = None):
        if not verbosity:
            return _NullTimer()
        return super().__new__(cls)

    def __init__(self, verbosity: bool = False, name: Optional[str] = None, logger: Optional[logging.Logger] = None):
        # verbosity already handled in __new__
        self.name = name or "Runtimer"
        self.logger = logger
        self._t0 = None
        self.last = 0.0
        self.total = 0.0
        self.count = 0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self._t0
        self.last = elapsed
        self.total += elapsed
        self.count += 1
        avg = self.total / self.count if self.count else 0.0
        msg = f"{self.name}: {elapsed:.6f}s (avg {avg:.6f}s over {self.count})"
        if self.logger is not None:
            try:
                self.logger.info(msg)
            except Exception:
                print(msg)
        else:
            print(msg)
        return False

    def elapsed(self) -> float:
        return self.last

    def __call__(self, fn: Callable):
        # decorator form
        def wrapper(*a, **kw):
            with self:
                return fn(*a, **kw)
        return wrapper
