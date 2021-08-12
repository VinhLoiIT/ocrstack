import setuptools

setuptools.setup(
    setup_requires=['pbr'],
    pbr=False,
    packages=setuptools.find_packages(exclude=['test']),
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.19',
        'matplotlib>=3.3',
        'editdistance',
        'pillow>=7',
        'pyyaml>=5',
        'torchinfo>=1.5',
    ],
)
