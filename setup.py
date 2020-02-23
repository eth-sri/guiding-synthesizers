from setuptools import setup, find_packages

setup(name='guidesyn',
      version='1.0',
      description='Guiding Program Synthesis by Learning to Generate Examples',
      author='Larissa Laich, Pavol Bielik',
      license='Apache License 2.0',
      install_requires=[
          'scikit-learn==0.21.3',
          'scikit-image >= 0.15',
          'pillow==6.2.0',
          'torch==1.2.0',
          'torchvision==0.4.2',
          'matplotlib==3.1.1',
          'tqdm==4.35.0',
          'Flask==1.1.1',
          'Flask-RESTful==0.3.7',
      ],
      packages=find_packages(),
   )
