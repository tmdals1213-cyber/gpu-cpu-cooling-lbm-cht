# Publishing and moving the package

Target repository: `tmdals1213-cyber/gpu-cpu-cooling-lbm-cht` (public).

Before committing:

```sh
python tools/verify_package.py --write-manifest
git status --short
git diff --cached --stat
```

Only the allowlisted source/setup/small-reference/media payload belongs in Git. `.gitignore` excludes generated fields, checkpoints, caches and environments. Do not force-add ignored files. Check staged paths before pushing; the package validator is a useful scan, not a guarantee that arbitrary future additions contain no private data.

If publishing manually with the official GitHub CLI:

```sh
gh auth login --hostname github.com --git-protocol https --web
gh repo create tmdals1213-cyber/gpu-cpu-cooling-lbm-cht --public --source . --remote origin --push
```

The create command assumes this is a new repository and the local tree has been committed. If the remote already exists, inspect it first and use a normal non-force push. Use the official browser/device authorization flow; do not paste passwords or access tokens into chat, source files or commit messages.

## Transfer to another local agent

Clone the public repository or build an offline bundle:

```sh
python tools/build_bundle.py
```

This creates `../dist/gpu-cpu-cooling-lbm-cht-portable.zip` and its checksum report. It includes the source, setup, tiny STL assets and summary media. It excludes `.git`, `runs/`, `.cache/` and virtual environments. Extract it anywhere, create a new Python environment, and start at `docs/AGENT_HANDOFF.md`.

The environment itself is not embedded: compatible Python, a GPU driver and pip dependencies must be installed on the destination workstation. New full calculations can take hours and produce gigabytes of local output.
