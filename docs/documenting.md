# Documenting

- Google-style docstrings without types; the signature carries the types. Every public function and class has one.
- Link API symbols with mkdocstrings cross-references: `` [`LPDO`][tn_noise_learning.LPDO] ``.
- Python code blocks in `docs/` and `README.md` run as tests (`just doctest`). A block that continues the previous one
  uses the brace form, which Material for MkDocs renders:

````markdown
```{.python continuation}
history = trainer.fit(model, train, test)
```
````

- `just docs` builds with `--strict` and link validation. The docs workflow also checks external links with lychee.
- Changes to the public API or to developer workflows update the docs, `README.md` and `AGENTS.md` in the same PR.
