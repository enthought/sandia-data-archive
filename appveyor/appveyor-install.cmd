rem install python packages
pip install --cache-dir C:/egg_cache nose
pip install --cache-dir C:/egg_cache coverage==3.7.1
pip install --cache-dir C:/egg_cache h5py
rem Work around bug in babel 2.0: see mitsuhiko/babel#174
pip install --cache-dir C:/egg_cache babel==1.3
pip install --cache-dir C:/egg_cache Sphinx

rem install sda
python setup.py develop
