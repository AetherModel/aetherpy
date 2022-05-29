Contributing
============

Code
----

Please read the [standards document](doc/design/standards) before
contributing.

### Setup

If you wish to contribute please start with checking out the `develop` branch
from your fork (you only need a fork if you don't have developer access to this
repository).

```sh
git clone <url to your fork here>
git checkout develop
```

Once you make your changes you can either test locally by importing `aetherpy`
into `ipython`:

```python
# assuming you are in the root directory of aetherpy
import aetherpy
```

Or you may install it to test how it may work globally while you continue to develop the code:

```python
python setup.py develop --user
```

### Development

Make new branches for features `git checkout -b my_feature` and commit often
and push a little less often. Try to merge back to main branch as soon as
you have something that works.

#### Linting

We recommend ensuring your code is follows PEP8 by using a linter such as
[flake8](https://flake8.pycqa.org/en/latest/).  This may by installed using pip.

To install `cpplint`

```sh
# depending on your system one of these lines applies
pip install --user flake8
pip install flake8
python3 -m pip flake8
python3 -m pip --user flake8
```

Using a linter in an editor is a good supplement, but not a replacement for
static linters.  The linter in the 'atom' editor requires that you install the
`linter` and `gcc-linter` packages.  Atom also has additional packages
`whitespaces` and `tabs-to-spaces` to automatically remove whitespaces at the
end of the lines, and convert tabs to spaces.

### Commit Styling

The first line of the commit must be *at most* ~50 characters long and
should start with either.

- `FEAT:` For new feature.
- `BUG:` For bug fix.
- `MERGE:` For merging.
- `DOC:` For documentation update.
- `TEST:` For the addition or modification of tests.
- `STY:` For a style update (e.g., linting).
- `DEP:` Deprecate something, or remove a deprecated object.
- `REVERT:` Revert an earlier commit.
- `MAINT:` For maintenance such as refactoring, typos, etc.

The commit first line must be in *present* tense so that the commit log has
consistent formatting. For more information check out [conventional commit
messages](https://www.conventionalcommits.org/en/v1.0.0/).

For example,

*do:*

```
FEAT: Hydrostatic density implementation
```

*don't:*

```
Implemented hydrostatic density. (feature)
```

### Pull Requests

Make sure you have linted and checked your code before asking for a pull
request. Before requesting a review, ensure the pull request check list has
been completed.  Another member must check the code and approve it before merge.

Issues
------

*Issues* are reporting bugs, feature requests, or goals for the project. In
order to submit an issue make sure it follows the [issue
template](.github/ISSUE_TEMPLATE).  Please search through the existing issues
before submitting a new one, as someone else may have already come accross and
reported the problem you've encountered.
