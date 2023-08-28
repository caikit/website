# Development Guide

This document describes how to set up development environments to begin contributing to caikit projects.

## Managing development environments with Tox

We use the virtual environment manager [tox](https://tox.wiki/en/4.10.0/) to run our builds. 
We think of it as Make, but for python.
Plenty of public documentation is available for tox, but the quick gist is that for every task, tox manages an isolated virtual environment for that task and those environments all reside in the `.tox` directory. 
These tasks and their environments are describe in the `tox.ini` file at the root of each repo.

## Installing tox

The `tox` tool must be installed on your system in order to run builds, since it is the entrypoint for all build commands.
Since `tox` is a python module, it may be installed in your favorite python virtual environment of choice and invoked from there.
Though it is important to note that the environment where you install tox will not be the environment where builds and tests are run.

You may also use a system package manager to install tox, e.g. on Mac you can run
```shell
brew install tox
```

### Building and testing

Running unit tests with tox should always be done against an installation of the library being built. 
This is the default behavior of tox, but can be disabled with the `skip_install` configuration to test against source code directly.

To run unit tests, the `py` environment will use your system's default version of python, and run pytest.
Other specific python version can be supplied in the format `py39`, `py310`, `py311`.
A python interpreter must be installed and available on your system path in order for tox to pick it up to create the virtual environment.
Example usage:
```shell
tox -e py311
```

The `[testenv]` section in your `tox.ini` file is used to specify the behavior for these test runs.
It should always allow passthrough args to the `pytest` invocation so that developers can pass args in, like:

```shell
tox -e py311 -- tests/some/package/test_file.py
```
```shell
tox -e py311 -- -k some_test_prefix
```

### Format and linting

The `fmt` environment should apply all code formatting rules. This should include at least `black` and `isort`.
The `lint` environment should apply all static linting, e.g. with `pylint`

Projects should define pre-commit hooks so that developers can enable pre-commit checks for formatting and linting.
Otherwise, train your fingers to
```shell
tox -e fmt,lint
```
before committing.

### Publishing a package

The `publish` environment should be set up to publish the package to PyPi.
We generally use [Flit](https://flit.pypa.io/en/stable/index.html) for this, simply because it's easy.

To test publication, publish to the [pypi test instance](https://test.pypi.org/) by setting `FLIT_INDEX_URL` to `https://test.pypi.org/simple/`.

To publish a package for the first time, a token will need to be used that has user-scoped permissions in order to create the project on pypi.
This is a bit dangerous! We generally create a short-lived token for that and add it to the github repo only long enough to run the initial publish.
Once the pypi package is created, we replace the user-scoped token with one that has permissions scoped to that specific package.

### Default Behavior

By default, the `tox` command should run
- Unit tests
- Code formatting
- Linting

For example the top of your `tox.ini` should look similar to:
```ini
[tox]
envlist = py, lint, fmt
```

### Specifying project dependencies

All dependencies should be specified in extras sets in the `pyproject.toml` file.
This allows other python environments to be easily built without requiring the use of `tox`.

These extras sets are specified in the `[project.optional-dependencies]` section of the pyproject.toml file, and should include at least:
- `dev-test`: Everything required to run unit tests
- `dev-fmt`: Everything required to run formatting and linting
- `dev-docs`: Everything required to build and publish documentation
- `dev-build` Everything required to build and publish the package itself

### How to define a new tox environment

New environments can be set up by adding another
```ini
[testenv:{env_name}]
```
section to the tox.ini file.

They should all include at least a `description`, `extras`, and `commands` field.

For example:
```ini
[testenv:joe_cool]
description = A super fun environment that prints hello world
extras = dev-test
commands = python -c 'print("hello world")'
```
is invoked as:
```shell
$ tox -e joe_cool 
joe_cool: commands[0]> python -c 'print("hello world")'
hello world
  joe_cool: OK (0.20=setup[0.17]+cmd[0.03] seconds)
  congratulations :) (0.27 seconds)
```

If environment variables are required, they must be specified in the `passenv` config.
If external scripts are invoked, they must be specified in the `allowlist_externals` config.
