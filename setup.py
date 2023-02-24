from setuptools import setup, find_packages

try:
  with open("README.md", "r") as readme:
    long_description = readme.read()
except:
  pass

setup(
    name='robust_control',
    version='0.3.0',
    author='Alexander Goryunov',
    author_email='alex.goryunov@gmail.com',
    description='A PyTorch imlpementation of the "robust" synthetic control algorithm',
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    url='https://github.com/agoryuno/robust_control/new/main',
    packages=["robust_control"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'torch',
    ],
)
