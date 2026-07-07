"""residency/ — gestione della residenza degli expert in VRAM + miss handling.

Il ResidencyManager decide, per ogni layer, quali expert stanno in VRAM e a che precisione, e cosa
fare quando un token è instradato su un expert non residente (i tre miss_mode, asse D).
"""

from msc.residency.manager import ResidencyManager, ExpertStore  # noqa: F401
from msc.residency.miss_modes import MissMode, MissHandler, make_miss_handler  # noqa: F401
