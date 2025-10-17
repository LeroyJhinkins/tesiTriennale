# setup.py

import os
from setuptools import setup, find_packages

# Create output directory if it doesn't exist
required_dirs = ['graphs','temp']
for dir in required_dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

setup(
    name="tesi_libs", # library names in the venv
    version="0.1",
    # Add libs
    packages=find_packages(),
    # Add custom commands
    entry_points={
        'console_scripts': [
            'run=utils.run_script:main',
        ],
    },
)
