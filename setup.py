from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

tests_require = [
  'pytest',
  'pytest-cov',
  'testfixtures',
]

setup(
  name='kviz',
  packages=['kviz'],
  version='0.0.11',
  description='A Library for visualizing keras neural networks',
  install_requires=[
    'tensorflow',
    'networkx',
    'pygraphviz',
    'matplotlib',
    'imageio'
  ],
  python_requires='>=3.9, <4',
  setup_requires=['pytest-runner'],
  tests_require=tests_require,
  extras_require={
    'test': tests_require,
  },
  long_description=long_description,
  long_description_content_type="text/markdown",
  license="Apache License 2.0",
  include_package_data=True,
  classifiers=[
    'Programming Language :: Python :: 3.9',
  ],
)
