from setuptools import setup

setup(name='novel_molecules_generation',
      version='0.1',
      description='New small molecules generation pipeline',
      url='http://github.com/',
      author='',
      author_email='',
      license='MIT',
      packages=['novel_molecules_generation', 'novel_molecules_generation.pipeline', 'novel_molecules_generation.ae', 'novel_molecules_generation.c2f',
                 'novel_molecules_generation.data_processing', 'novel_molecules_generation.e2f', 'novel_molecules_generation.seqtoseq'
                ],
      zip_safe=False)