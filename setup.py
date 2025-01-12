import setuptools

with open('README.rst', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

with open('VERSION', 'r') as fh:
    version = fh.read().strip()

setuptools.setup(name='ndlar_flow',
                 version=version,
                 description='An h5flow-based analysis framework for Module0 data',
                 long_description=long_description,
                 long_description_content_type='text/x-rst',
                 author='Peter Madigan',
                 author_email='pmadigan@berkeley.edu',
                 package_dir={'': 'src'},
                 python_requires='>=3.7',
                 install_requires=[
                     'h5py>=2.10',
                     'pytest',
                     'scipy',
                     'scikit-image',
                     'scikit-learn>=1.3.0',
                     'h5flow>=0.2.0',
                     'pylandau @ git+https://github.com/cuddandr/pylandau.git#egg=pylandau',
                     'adc64format @ git+https://github.com/larpix/adc64format.git@v0.1.2#egg=adc64format',
                 ]
                 )
