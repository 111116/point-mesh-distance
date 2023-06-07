from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='point_mesh',
      ext_modules=[
          cpp_extension.CppExtension('point_mesh', ['point2mesh.cpp', 'point2mesh_kernel.cu']),
      ],
      cmdclass={
          'build_ext': cpp_extension.BuildExtension
      })
