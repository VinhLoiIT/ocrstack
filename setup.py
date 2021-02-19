import setuptools

setuptools.setup(
    setup_requires=['pbr'],
    pbr=True,
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.19',
        'matplotlib>=3.3',
        'editdistance',
        'pillow>=7',
        'pyyaml>=5.4.1',
    ],
)
