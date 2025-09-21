# project/runner/logging_filters.py
from __future__ import annotations
import sys, io, contextlib

class DevNull:
    def __init__(self): self.encoding = "utf-8"
    def write(self, s): return len(s)
    def flush(self): pass
    def isatty(self): return False

@contextlib.contextmanager
def silence_stdio(also_stderr=True):
    saved_out, saved_err = sys.stdout, sys.stderr
    try:
        sys.stdout = DevNull()
        if also_stderr:
            sys.stderr = DevNull()
        yield
    finally:
        sys.stdout = saved_out
        if also_stderr:
            sys.stderr = saved_err

class LineFilterWriter:
    def __init__(self, underlying, patterns):
        self._u = underlying
        self._buf = io.StringIO()
        self._pat = tuple(patterns)
    def write(self, s):
        self._buf.write(s)
        text = self._buf.getvalue()
        if "\n" not in text:
            return len(s)
        lines = text.splitlines(keepends=True)
        if not text.endswith("\n"):
            self._buf = io.StringIO(); self._buf.write(lines[-1]); lines = lines[:-1]
        else:
            self._buf = io.StringIO()
        kept = [ln for ln in lines if not any(p in ln for p in self._pat)]
        if kept:
            self._u.write("".join(kept))
        return len(s)
    def flush(self):
        tail = self._buf.getvalue()
        if tail and not any(p in tail for p in self._pat):
            self._u.write(tail)
        self._u.flush()

@contextlib.contextmanager
def mute_logs(patterns=("[kgr:year]",), enabled=True, streams=("stdout", "stderr")):
    if not enabled:
        yield; return
    saved = {}
    try:
        import sys
        if "stdout" in streams:
            saved["stdout"] = sys.stdout
            sys.stdout = LineFilterWriter(sys.stdout, patterns)
        if "stderr" in streams:
            saved["stderr"] = sys.stderr
            sys.stderr = LineFilterWriter(sys.stderr, patterns)
        yield
    finally:
        if "stdout" in saved:
            try: sys.stdout.flush()
            except Exception: pass
            sys.stdout = saved["stdout"]
        if "stderr" in saved:
            try: sys.stderr.flush()
            except Exception: pass
            sys.stderr = saved["stderr"]

def mute_kgr_year_logs_if(*, no_life_table: bool):
    return mute_logs(patterns=("[kgr:year]",), enabled=bool(no_life_table))
