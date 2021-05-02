from setuptools import setup, find_packages

setup(name='STAM',
version='0.1',
author='Gilad Sharir',
author_email='gilad.sharir@alibaba-inc.com',
packages=find_packages(),
install_requires=[
    "torch",
    "torchvision",
    "numpy",
    "pillow",
    "av"
],
zip_safe=False,
)