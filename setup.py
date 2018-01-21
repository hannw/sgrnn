from setuptools import setup, find_packages

setup(name='sgrnn',
      version='0.1',
      description='synthetic gradient for rnn.',
      url='https://github.com/hannw/sgrnn',
      author='Hann Wang',
      author_email='hann@ucla.edu',
      license='',
      packages=find_packages(),
      tests_require=['nose'],
      install_requires=['numpy']
      )