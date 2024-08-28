from setuptools import setup, find_packages

setup(
    name='DLOlympus',
    version='0.1.0',
    author='Facundo Roffet',
    author_email='facundo.roffet@cs.uns.edu.ar',
    url='https://github.com/FacuRoffet99/DLOlympus',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'fastai==2.7.14',
    ],
)