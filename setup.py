# -*- coding: utf-8 -*-
""" install script
    thanks to Kenneth Reitz https://github.com/kennethreitz
    for publishing this setup.py template
"""
# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
from shutil import rmtree

from setuptools.command.install import install
from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'ksu'
DESCRIPTION = 'Implementation of the KSU compression algorithm https://www.cs.bgu.ac.il/~karyeh/compression-arxiv.pdf'
URL = 'https://github.com/nimroha/ksu_classifier'
EMAIL = 'nimrod.morag@gmail.com'
AUTHOR = 'Nimrod Morag, Yuval Nissan'
REQUIRES_PYTHON = '>=2.7.0'
VERSION = None
REQUIRED = None

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec (f.read(), about)
else:
    about['__version__'] = VERSION

if not REQUIRED:
    with io.open(os.path.join(here, 'requirements.txt')) as f:
        REQUIRED = f.read().split('\n')

class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPi via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()

def cloneSubmodules(): #TODO only works in project root dir (TODO doesn't seem to clone)
    os.system('git submodule init')
    os.system('git submodule update --recursive')  # get the epsilon-net creation module from github

class CustomInstallCommand(install):
    """Add custom functionality to install command."""

    @staticmethod
    def _called_from_setup(run_frame): # hack to force backward-compatible install
        return True

    def run(self):
        self.execute(cloneSubmodules, (), msg='Fetching 1-nn-condensing submodule')
        install.run(self) # run regular install

# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['ksu'],

    entry_points={
        'console_scripts': ['ksu=ksu.EntryPoint:main'],
    },
    # scripts=['ksu/EntryPoint.py'],

    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Development Status :: 4 - Beta'
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
        # 'install': CustomInstallCommand,
    },
)




