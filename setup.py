from setuptools import setup, find_packages

setup(
    name="image_processor",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
    ],
)