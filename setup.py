import setuptools

setuptools.setup(
    name="xopr",
    version="0.0.1",
    author="Megvii Engine Team",
    author_email="megengine@megvii.com",
    description="Experimental operator library for MegEngine. More fashion opr, basicly pure python and compatiable with torch",
    packages=setuptools.find_packages(where="src"),
    long_description="""
        # xopr
        ``xopr`` is an open-source Python package. Experimental operator package for MegEngine
        More fashion operators, basicly pure python and compatiable with torch
        ## Installation
        ``xopr`` is easy to install with pip:
        ```
        pip install xopr
        ```
    """,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    include_package_data=True,
    url="https://github.com/MegEngine/xopr",
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="megengine deep learning",
    python_requires=">=3.6.1",
    install_requires=["megengine>=1.8.0"],
)
