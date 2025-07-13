from pathlib import Path

from setuptools import find_packages, setup


def read_long_description() -> str:
    readme_file = Path(__file__).with_name("README.md")
    return (
        readme_file.read_text(encoding="utf-8")
        if readme_file.exists()
        else "Synthetic text & equation image generator."
    )


setup(
    name="text2image",
    version="0.1.0",
    description="Utilities for generating synthetic images of text or LaTeX equations.",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="Abdulbasit Zahir",
    python_requires=">=3.8",
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=[
        "pillow>=9.0.0",
        "numpy>=1.20",
        "opencv-python>=4.6",
        "matplotlib>=3.5",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux, MacOS",
    ],
    include_package_data=True,
)
