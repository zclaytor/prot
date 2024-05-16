import setuptools
import codecs
import os.path


# These two functions are just to read the version number. Sourced from
# https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name="prot",
    version=get_version("prot/__init__.py"),
    author="Zachary R. Claytor",
    author_email="zclaytor@stsci.edu",
    install_requires=["lightkurve"],
    packages=setuptools.find_packages(),
    include_package_data=True,
)