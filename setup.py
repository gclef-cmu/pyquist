from setuptools import setup

setup(
    name="pyquist",
    packages=["pyquist"],
    install_requires=["numpy", "resampy", "soundfile", "sounddevice"],
)
