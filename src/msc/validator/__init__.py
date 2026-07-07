"""validator/ — segnale binario corretto/non-corretto, a CONTESTO CRESCENTE.

È il cuore (docs/00_architecture.md §8). Due famiglie di task + un riempitore di contesto che
porta ogni caso alle lunghezze target {1k,4k,16k,64k,max} e registra l'accuratezza PER lunghezza.
"""

from msc.validator.base import Validator, ValItem, ValVerdict, ContextLengthResult  # noqa: F401
from msc.validator.python_unit_tests import PythonUnitTestValidator  # noqa: F401
from msc.validator.closed_form import ClosedFormValidator, NeedleInHaystackValidator  # noqa: F401
from msc.validator.context_filler import ContextFiller, FillStrategy  # noqa: F401
