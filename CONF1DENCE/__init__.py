import os as _os

try:
    from ._version import __version__
except(ImportError):
    pass

               
from . import utils
from . import train
from . import predict      

# Absolute path to the repository root (the directory that contains both the
# ``CONF1DENCE`` package and the ``data`` folder). Derived from this file so it
# stays correct regardless of where the repository is checked out.
__abspath__ = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))) + '/'
