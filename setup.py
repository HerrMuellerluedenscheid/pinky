import sys

from setuptools import setup


packname = 'pinky'
version = '0.1'


setup(
    name=packname,
    version=version,
    license='GPLv3',
    python_requires='!=3.0.*, !=3.1.*, !=3.2.*, <4',
    install_requires=[],
    packages=[packname],
    package_dir={'pinky': 'src'},
    entry_points={
        'console_scripts': [
            'pinky = pinky.model:main',
        ]
    }
)
