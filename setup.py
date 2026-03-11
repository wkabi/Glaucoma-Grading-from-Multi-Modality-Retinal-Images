from setuptools import setup, find_packages

setup(
    name="optic",
    version="0.1",
    author="Bingyuan Liu",
    description="For GAMMA challenge",
    packages=find_packages(),
    python_requries=">=3.8",
    install_requires=[
        # Please install pytorch-related libraries and opencv by yourself based on your environment
    ],
)
