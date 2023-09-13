from setuptools import setup, find_packages

setup(
    name='cubeml',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'tensorboard',
        'scikit-learn',
        'scipy',
        'deap',
        'matplotlib',
        'seaborn',
        'pandas',
        'Pillow',
        'cubeml'
    ],
    author='Michael Nagle',
    author_email='michael.nagle@oregonstate.edu',
    description='Semantic segmentation of hyperspectral images (hypercubes) with machine learning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/naglemi/cubeml', 
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
