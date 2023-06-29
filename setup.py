from setuptools import setup, find_packages

with open("requirements.txt") as requirement_file:
    requirements = requirement_file.read().split()

setup(
    name="video_analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
)


