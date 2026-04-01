# MINA

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/saezlab/MINA/test.yaml?branch=main
[badge-docs]: https://img.shields.io/badge/docs-GitHub%20Pages-blue

## Multicellular INtegration Analysis
`mina` provides a bridge between single-cell data analysis workflows from `scverse`, factor based models from `MOFAflex`, and prior knowledge to generate tissue-centric descriptions from single-cell data.

This package facilitates the implementation of [Multicellular Factor Analysis](https://elifesciences.org/articles/93161) by providing functions to process and format single-cell data into a multi-view format, together with additional visualization and downstream tasks to analyse and interpret multicellular programs.

## Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv].
MINA currently targets `mofaflex==0.1.0.post1`.

There are several alternative options to install MINA:

<!--
1) Install the latest release of `MINA` from [PyPI][]:

```bash
pip install MINA
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/saezlab/MINA.git@main
```

## Citation

> Ricardo Omar Ramirez Flores, Jan David Lanzer, Daniel Dimitrov, Britta Velten, Julio Saez-Rodriguez (2023) Multicellular factor analysis of single-cell data for a tissue-centric understanding of disease eLife 12:e93161

[uv]: https://github.com/astral-sh/uv
[tests]: https://github.com/saezlab/MINA/actions/workflows/test.yaml
[documentation]: https://saezlab.github.io/mina/
