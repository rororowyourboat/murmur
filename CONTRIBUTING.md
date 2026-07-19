# Contributing to Murmur

Thanks for helping improve Murmur. The workflow is intentionally small:

1. Search the issues, then open one for a bug or meaningful feature.
2. Create a short-lived branch from `main` (for example, `fix/recording-stop`).
3. Keep the change focused and add or update tests when behavior changes.
4. Run `make check`.
5. Open a draft pull request and link its issue with `Closes #123`.
6. Mark it ready once CI passes and all review conversations are resolved.

Small documentation and typo fixes may skip the issue. Please do not include
unrelated cleanup in the same pull request.

## Development setup

Murmur uses Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
make install
make check
```

Use clear, imperative commit messages. Pull requests are squash-merged, so a
perfectly curated commit history is not required.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
For vulnerabilities, follow [SECURITY.md](SECURITY.md) instead of opening a
public issue.
