import pytest
import pytest_cov

# import extra deps and use it to keep pipreqs and flake8 happy
for pkg in (pytest, pytest_cov):
    print("%s %s" % (pkg.__name__, pkg.__version__))
