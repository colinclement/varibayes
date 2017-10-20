from setuptools import setup

setup(name='varibayes',
      license='MIT License',
      author='Colin Clement',
      author_email='colin.clement@gmail.com',
      url='https://github.com/colinclement/varibayes',
      version='0.0.1',
      platforms='osx, posix, linux, windows',
      install_requires=[
          "numpy>=1.10.4"
      ],

      packages=['varibayes'],

      description='A toolbox for performing variational Bayesian inference',
      long_description="""\
        Variational Bayesian Inference Toolbox
        
        --------------------------------------


        This module is inspired by the paper 'Black Box Variational Inference'
        by Rajesh Ranganath et al. It attempts to make nearly trivial the task
        of fitting a variational distribution to a user-specified log-likelihood
        function without derivatives. Currently it only uses a
        mean field variational distribution, but the main class 
        VariationalInferenceMF is flexible enough for simple subclassing in the
        future. This module also contains a number of implementations of
        stochastic gradient descent algorithms to be used for optimization.
      """,

      classifiers=[
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'Development Status :: 2 - Pre-Alpha',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Scientific/Engineering :: Mathematics',
      ],
)
