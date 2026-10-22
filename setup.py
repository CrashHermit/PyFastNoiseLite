import os
import platform
from setuptools import Extension, find_namespace_packages, setup
from Cython.Build import cythonize

__author__ = "Tiziano Bettio"
__license__ = "MIT"
__version__ = "0.0.8.dev2"

__copyright__ = """
Copyright (c) 2021 Tiziano Bettio
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

with open("README.md") as fh:
    long_description = fh.read()

with open("VERSION") as fh:
    version = fh.read().strip()

if platform.system() == "Windows":
    EXTRA_COMPILE_ARGS = []
    EXTRA_LINK_ARGS = []
elif platform.system() == "Darwin":
    EXTRA_COMPILE_ARGS = ["-std=c++11", "-stdlib=libc++"]
    EXTRA_LINK_ARGS = ["-std=c++11", "-stdlib=libc++"]
else:
    EXTRA_COMPILE_ARGS = ["-std=c++11"]
    EXTRA_LINK_ARGS = ["-std=c++11"]

# Add "c++_shared" to libraries if installed via python-for-android
LIBRARIES = []
if "ARCH" in os.environ and os.environ["ARCH"].startswith("arm"):
    LIBRARIES.append("c++_shared")

extension = Extension(
    "pyfastnoiselite.pyfastnoiselite",
    ["src/pyfastnoiselite/pyfastnoiselite.pyx"],
    include_dirs=["ext/FastNoise", "ext/CPPWrapper"],
    extra_compile_args=EXTRA_COMPILE_ARGS,
    extra_link_args=EXTRA_LINK_ARGS,
    language="c++",
    libraries=LIBRARIES,
)

extensions = cythonize(
    [extension],
    compiler_directives={"language_level": 3, "embedsignature": True},
    annotate=False,
)


setup(
    name="pyfastnoiselite",
    version=version,
    author="Tiziano Bettio",
    author_email="tc@tizilogic.com",
    description="Cython wrapper for Auburns' FastNoise Lite.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tizilogic/pyfastnoiselite",
    packages=find_namespace_packages(where="src"),
    package_data={"pyfastnoiselite": ["LICENSE", "VERSION", "pyfastnoiselite.pyi"]},
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.13",
    install_requires=["numpy"],
    ext_modules=extensions,
)
