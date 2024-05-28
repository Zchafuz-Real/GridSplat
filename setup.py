from setuptools import setup, find_packages

setup(
    name='GridSplat',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'numpy',
        'viser',
        'gsplat',
        'pytorch_msssim',
        'imageio',
        'torchvision',
        'Pillow',
        'matplotlib'
    ],
    entry_points={
        'console_scripts': [
            'start3d=start3d:main'
        ]
    }
)