from setuptools import find_packages, setup

from dict_minimize import __version__


def read_requirements(name):
    with open("requirements/" + name + ".in") as f:
        requirements = f.read().strip()
    requirements = requirements.replace("==", ">=").splitlines()  # Loosen strict pins
    return [pp for pp in requirements if pp[0].isalnum()]


requirements = read_requirements("base")
framework_requirements = read_requirements("frameworks")
test_requirements = read_requirements("tests")

with open("README.rst") as f:
    long_description = f.read()

setup(
    name="dict_minimize",
    version=__version__,
    packages=find_packages(),
    url="https://github.com/twitter/dict_minimize/",
    author="Ryan Turner",
    author_email=("rdturnermtl@github.com"),
    license="Apache v2",
    description="Access scipy optimizers from your favorite deep learning framework.",
    python_requires=">=3.6",
    install_requires=requirements,
    extras_require={"framework": framework_requirements, "test": test_requirements},
    long_description=long_description,
    long_description_content_type="text/x-rst",
    platforms=["any"],
)
