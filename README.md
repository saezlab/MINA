# MINA

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/saezlab/MINA/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/MINA

Multicellular INtegration Analysis

## Getting started
`mina` provides a bridge between single-cell data analysis workflows from `scverse`, factor based models from `MOFAflex`, and prior knowledge to generate tissue-centric descriptions from single-cell data.

This package facilitates the implementation of [Multicellular Factor Analysis](https://elifesciences.org/articles/93161) by providing functions to process and format single-cell data into a multi-view format, together with additional visualization and downstream tasks to analyse and interpret multicellular programs.

## Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv].

There are several alternative options to install MINA:

<!--
1) Install the latest release of `MINA` from [PyPI][]:

```bash
pip install MINA
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/saezlab/MINA.git@refact_dev
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/saezlab/MINA/issues
[tests]: https://github.com/saezlab/MINA/actions/workflows/test.yaml
[documentation]: https://MINA.readthedocs.io
[changelog]: https://MINA.readthedocs.io/en/latest/changelog.html
[api documentation]: https://MINA.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/MINA
