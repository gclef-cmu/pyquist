from setuptools import setup

setup(
    name="pyquist",
    packages=["pyquist", "pyquist.web"],
    install_requires=[
        "numpy~=2.0.2",
        "resampy",
        "soundfile",
        "sounddevice",
        "tqdm",
        "requests",
        "llvmlite>=0.43.0",
        "tqdm",
    ],
)
