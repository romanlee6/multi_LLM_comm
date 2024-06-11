import setuptools

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name='gym_dragon',
    version='0.2.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(include=['gym_dragon']),
    python_requires='>=3.9',
    install_requires=[
        'gym==0.21.0',
        'matplotlib',
        'numpy>=1.23.3',
        'ray[rllib]>=2',
    ],
)
