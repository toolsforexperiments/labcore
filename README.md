# labcore
Code for measuring and such.

## installation
- clone from github
- make the right environment: use ``$ conda env create -n labcore --file environment.yml``
- install into your measurement environment:
  ``$ pip install -e <path_to_cloned_repository>``
- you should then be able to import:
  ``>>> import labcore``
  
## requirements
what other packages are needed will depend a lot on which tools from this package you'll be using.
In general, this will be used in a typical qcodes measurement environment.
The different submodules in this package should list in their documentation if additional packages are required.
