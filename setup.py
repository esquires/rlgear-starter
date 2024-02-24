from setuptools import find_packages, setup

setup(
    name="rlgear_starter",
    version="0.0.1",
    author="Eric Squires",
    long_description="",
    description="",
    zip_safe=False,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # these are the versions of the software when this file was created
        "ray[tune]==2.9.3",
        "gymnasium==0.29.1",
        "tensorboard",
        "kaleido",
    ],
)
