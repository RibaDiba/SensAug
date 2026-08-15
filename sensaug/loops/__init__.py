"""Custom MMEngine train/val/test loops, one module per concern.

| module              | what lives there                                          |
|---------------------|-----------------------------------------------------------|
| `train_loops.py`    | loops a real training job runs                            |
| `test_loops.py`     | development/inspection loops -- no metrics, no training   |
| `sensaug_loop.py`   | the sensitivity-analysis pipeline (`--aug-type=ours`)     |
| `grad_corr_loop.py` | the cross-correlation pipeline (`--aug-type=grad_corr`)   |

Importing this package registers every loop with the mmseg LOOPS registry, which
is what `train.py`'s `from sensaug.loops import *` is for.

Deliberately NOT wrapped in `try/except ImportError` the way
`sensaug/hooks/__init__.py` is. A swallowed import here would not fail loudly --
it would silently un-register a loop, and the failure would resurface much later
and much less legibly as `KeyError: 'RobustValLoop is not in the mmseg::loop
registry'`.
"""

from .train_loops import *  # noqa: F401,F403
from .test_loops import *  # noqa: F401,F403
from .sensaug_loop import *  # noqa: F401,F403
from .grad_corr_loop import *  # noqa: F401,F403

from .train_loops import __all__ as _train_all
from .test_loops import __all__ as _test_all
from .sensaug_loop import __all__ as _sensaug_all
from .grad_corr_loop import __all__ as _grad_corr_all

__all__ = [*_train_all, *_test_all, *_sensaug_all, *_grad_corr_all]
