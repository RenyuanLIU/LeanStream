import os
from backends.triton.mapped import Mapped, library as _library

def library(unused):
    return _library(os.environ["LEANSTREAM_MAPPED_LIBRARY"])
