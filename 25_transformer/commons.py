# Source - https://stackoverflow.com/a/30024601
# Posted by gojomo, modified by community. See post 'Timeline' for change history
# Retrieved 2026-09-30, License - CC BY-SA 3.0

from contextlib import contextmanager
from timeit import default_timer

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start
