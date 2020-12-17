from setuptools import setup, find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='Trolldetector',
    version='1.0',
    description='This command line tool is used to apply five different classification techniques on a dataset '
                'consisting of troll and nontroll tweets.',
    author='Robin Kösters',
    author_email='robin.koesters@uni-duesseldorf.de',
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": ["trolldetector=trolldetector.main:main"]},
    install_requires=requirements,
)