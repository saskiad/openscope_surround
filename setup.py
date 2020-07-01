import setuptools

setuptools.setup(
    name='openscope',
    description='Tools for surround-suppression AIBS Open Scope project.',
    packages=['oscopetools'],
    install_requires=[
        'psychopy',
        'numpy',
        'pandas',
        'matplotlib',
        'h5py',
        'pillow',
        'nd2reader',
        'pg8000',
    ],
)
