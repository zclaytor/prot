import setuptools

# Load the __version__ variable without importing the package already
exec(open("prot/version.py").read())

setuptools.setup(
    name="prot",
    version=__version__,
    author="Zachary R. Claytor",
    author_email="zclaytor@hawaii.edu",
    install_requires=["lightkurve"],
    packages=setuptools.find_packages(),
    include_package_data=True,
)