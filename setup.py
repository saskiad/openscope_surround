import setuptools

setuptools.setup(
    name='openscope',
    description='Tools for surround-suppression AIBS Open Scope project.',
    packages=['analysis'],
    install_requires=[
        'psychopy',
        'numpy',
        'pandas',
        'matplotlib',
        'h5py',
        'pillow',
        'nd2reader',
    ]
)
