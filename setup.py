from setuptools import setup

setup(
    name="pyquist",
    packages=["pyquist", "pyquist.web"],
    install_requires=["numpy", "resampy", "soundfile", "sounddevice", "tqdm"],
)
