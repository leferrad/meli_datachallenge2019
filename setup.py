from setuptools import setup, find_packages

setup(
    name='melidatachall19',
    version='0.1.0',
    author='Leandro Ferrado',
    author_email="leferrad@gmail.com",
    packages=find_packages(
        exclude=['data', 'docs', 'notebooks', 'tests', 'tools']
    ),
    license='MIT',
    description='A Python solution for the MercadoLibre Data Challenge 2019',
    long_description=open('README.md').read(),
    install_requires=open('requirements.txt').read().split(),
    extras_require={
        'tests': [
            'pytest==6.2.0',
            'pytest-pep8==1.0.6',
            'pytest-cov==2.10.1',
            'pytest-bdd==4.0.2',
        ],
    }
)
