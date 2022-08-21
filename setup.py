from configparser import ConfigParser

import setuptools
from pkg_resources import parse_version

assert parse_version(setuptools.__version__) >= parse_version("36.2")

# note: all settings are in settings.ini; edit there, not here
config = ConfigParser(delimiters=["="])
config.read("setup.cfg")
cfg = config["SETUP"]

py_versions = "3.7 3.8 3.9".split()
min_python = cfg["min_python"]

requirements = ["pip", "packaging"]
if cfg.get("requirements"):
    requirements += cfg.get("requirements", "").split()

setuptools.setup(
    name=cfg["lib_name"],
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=" + cfg["min_python"],
    zip_safe=False,
    version=cfg["version"],
    entry_points={"console_scripts": cfg.get("console_scripts", "").split()},
)
