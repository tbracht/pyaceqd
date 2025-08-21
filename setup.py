from setuptools import setup, find_packages

setup(
    name='pyaceqd',
    version='0.0.1',
    packages=find_packages(),
    package_data={
        'pyaceqd.two_time': ['*.so'],  # This will include the compiled .so file
    },
    install_requires=[
        'numpy',
        'matplotlib',
        'tqdm',
        'scipy'
    ],
)