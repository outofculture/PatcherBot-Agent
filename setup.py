"""Setup script for the PatcherBot-Agent package."""

from __future__ import annotations

from pathlib import Path
from setuptools import find_packages, setup


def parse_requirements(filename: str) -> list[str]:
    """
    Load requirements from a pip requirements file.
    
    Args:
        filename (str): The relative path to the requirements file.

    Returns:
        list[str]: A list of dependency strings suitable for install_requires.

    Notes:
        - Ignores empty lines and lines starting with '#' (comments).
    """
    req_path = Path(__file__).parent / filename
    with req_path.open() as req_file:
        return [
            line.strip()
            for line in req_file
            if line.strip() and not line.startswith("#")
        ]


setup(
    name="PatcherBot-Agent",
    version="0.1",
    description="Deep Learning guided Automated Patch Clamp Electrophysiology System",
    url="https://github.com/PBL-gatech/PatcherBot-Agent",
    author="Benjamin Magondu, Nathan Malta, Kaden StillWagon, Victor Guyard",
    author_email="bmagondu3@gatech.edu",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(),
    install_requires=parse_requirements("requirements/requirements_runtime.txt"),
    extras_require={
        "acq4-detectors": [
            "torch>=2.2", "torchvision>=0.17", "timm>=1.0.20",
            "diffusers==0.40.0", "transformers>=4.56",
            "opencv-python>=4.8", "Pillow>=9.5",
        ],
        "acq4-continuous": ["nidaqmx"],
        # Imports of the PBL-gatech/robomimic clone set as policy.robomimicRoot (not pip-installed).
        "acq4-policies": ["termcolor", "transformers>=4.56"],
        "acq4-pco": ["pco"],
    },
)

