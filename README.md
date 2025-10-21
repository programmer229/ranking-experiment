# tufa labs template

tufa-labs reference template for maintainable python projects

## Installation

clone the project and navigate to the root, then run:

```bash
uv sync
source .venv/bin/activate
```

### Development

when you first clone the repo and you intend to push changes run the following:

```bash
pre-commit install
```

if you want to run pre-commit hooks manually run:

```bash
pre-commit run --all-files
```

## Other Notes

### General Repo Structure

In general, code should be organized as a
[python package](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
where there are two different approaches ("src layout" and "flat layout"),
and it is up to personal preference which one to use.

### Adding Vendored Dependencies

If you want to vendor important dependencies add them as a git submodule, i.e.:

```bash
git submodule add git@github.com:jax-ml/jax.git my-vendored-submodule
```

then add them to `uv` with

```bash
uv add ./my-vendored-submodule
```
