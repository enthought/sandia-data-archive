sandia-data-archive
===================

Tools for reading, writing, altering, and inspecting Sandia Data Archive (SDA)
files.

.. image:: https://api.travis-ci.org/enthought/sandia-data-archive.png?branch=master
   :target: https://travis-ci.org/enthought/sandia-data-archive
   :alt: Build status

.. image:: https://ci.appveyor.com/api/projects/status/fbg3ut4bggrevalf/branch/master?svg=true
   :target: https://ci.appveyor.com/project/EnthoughtOSS/sandia-data-archive
   :alt: Build status (Windows)

.. image:: https://coveralls.io/repos/github/enthought/sandia-data-archive/badge.svg?branch=master
   :target: https://coveralls.io/github/enthought/sandia-data-archive?branch=master
   :alt: Coverage status


Installation
------------

The source is hosted on GitHub at
https://github.com/SMASHtoolbox/SDAlibrary under the ``python`` directory.
Releases are available on the `Python package index
<https://pypi.python.org/pypi/sdafile>`_. The package can be installed via pip:
``pip install sdafile``.


Installation from source
------------------------

After downloading the source from GitHub, issue the following command from the
command line within the ``python`` directory::

    python setup.py install

To install the package in development mode, instead issue the command::

    python setup.py develop


Releasing to PyPI
-----------------

To make a new release of the package, following these steps.

Update Version
~~~~~~~~~~~~~~

Edit the file ``sdafile/version.py`` to update the ``dev`` flag on the version
number. For releases, ``dev`` should be ``False``.

Once ``sdafile/version.py`` is edited, upload it to the ``master`` branch on
GitHub.

Release on GitHub (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To help track the history of the package, `release the package on GitHub
<https://help.github.com/articles/creating-releases/>`_.

Upload to PyPI
~~~~~~~~~~~~~~

Package maintainers can make project releases to PyPI by running the command ::

    python setup.py sdist upload -r pypi

The PyPI upload system is described more completely `here
<http://peterdowns.com/posts/first-time-with-pypi.html>`_.

Update Version (Again)
~~~~~~~~~~~~~~~~~~~~~~

Now update the project version number. The version should get a new minor
version bump. Be sure to follow `semantic versioning practices
<https://semver.org/>`_. For non-releases, the ``dev`` flag should be ``True``.
Again, upload the file to the ``master`` branch on GitHub.


Building Documentation
-----------------------

To build documentation, you'll need a python environment with ``sphinx`` and
``sphinxcontrib-napoleon`` installed. (Both are available via pip.) To build
PDF documentation, you'll also need LaTeX on your machine. 

Documentation is found in the ``docs`` subdirectory. In that directory, you can
run ``make help`` to view all the build targets and instructions for building
them. Built documentation appears in the ``_build`` subdirectory.


Dependencies
------------

- `H5Py <http://www.h5py.org>`_
- `NumPy <http://www.numpy.org>`_
- `SciPy <http://www.scipy.org>`_
- `Pandas <http://pandas.pydata.org>`_


License
-------
`BSD3 <LICENSE>`_
