from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="cuda_utils",
    ext_modules=[
        CUDAExtension(
            name="cuda_utils._C",
            sources=[
            "cuda_utils.cu",
            "map_process.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)))]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
