from setuptools import setup

setup(
    name='tank-saturdays',
    description='Gym environment that will be used in a competition between the members of the Reinforcement Learning track form the 3rd Edition of AI Saturdays Madrid.',
    url='https://github.com/miguel-bm/tank-saturdays',
    author='miguel-bm',
    author_email='miguel.blanco.marcos@gmail.com',
    version='0.1',
    install_requires=[
        'numpy>=1.17.2', 'recordclass>=0.13.2', 'gym>=0.15.6', 'typer>=0.0.10',
        ],
    #scripts=['play_tanks.py'],
)
