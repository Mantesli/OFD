#!/usr/bin/env python
"""
Setup script for oilfield-leak-detection package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ""

# 读取requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip() 
        for line in requirements_path.read_text().split('\n')
        if line.strip() and not line.startswith('#')
    ]
else:
    requirements = []

setup(
    name="oilfield-leak-detection",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-modal oil field leak detection using RGB and infrared imagery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/oilfield-leak-detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "oilfield-sample=scripts.sample_frames:main",
            "oilfield-validate=scripts.validate_clip:main",
            "oilfield-extract=scripts.extract_features:main",
        ],
    },
)
