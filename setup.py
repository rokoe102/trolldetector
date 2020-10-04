from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='Trolldetector',
    version='0.1',
    description='Just an example',
    author='Robin Kösters',
    author_email='robin.koesters@uni-duesseldorf.de',
    packages=["trolldetector"],
    entry_points={
        "console_scripts": ["trolldetector=trolldetector.main:main"]},
    install_requires=requirements,
)