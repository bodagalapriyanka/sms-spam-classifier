# Files to publish

## Include

- Python source and tests.
- README, learning guide, learning journey, project plan, and documentation.
- `requirements.txt`, `requirements-lock.txt`, and `.gitignore`.
- The public UCI dataset, its original archive/documentation, and teaching/diagnostic examples under `data/`.
- The two synthetic challenge reports under `reports/`, with their documented provenance limitations.
- A code license and CI workflow once added.

The retained public source archive is intentional: the data guide references it. The synthetic predictions are intentional learning artifacts. Review any new datasets and reports before including them.

## Exclude

`.gitignore` excludes virtual environments, Python caches, generated models and their local metrics, common credential files, local environment settings, editor settings, logs, and build outputs. Keep private datasets in `data/private/` or `data/local/`; both directories are ignored. Do not place personal messages into the public `data/messages.csv`.

Ignore rules recognize filenames and paths, not sensitive content. A credential or private message embedded in an otherwise publishable file still needs to be removed before committing.

Never include API keys, passwords, OAuth tokens, database connection secrets, private SSH keys, or cloud credentials. Share configuration templates only with placeholder values; `.env.example` is allowed for that purpose and must be reviewed too.

Review large datasets, videos, archives, and other binary files individually. Prefer a documented download source or an appropriate artifact store when they do not belong in source control; consider Git LFS for necessary versioned large assets. The current UCI source ZIP is about 200 KB and is deliberately retained with its documentation. There is no blanket archive exclusion because it would hide that intended source artifact.

## Review before the first push

After Git is initialized, inspect ignored and untracked files:

```sh
git status --short --ignored
```

Stage the reviewed files, then check the actual commit contents:

```sh
git diff --cached --name-only
git diff --cached
```

Check for credentials, private messages, unexpected large files, and local machine paths in newly generated reports. `.gitignore` does not remove files already tracked by Git; if something was previously committed, it requires separate cleanup.

Publish to the owner's GitHub account first. For a later independent repository, prepare a reviewed source package without `.git`, ignored local files, or credentials. Keep code authorship, license terms, and dataset attribution accurate when sharing that package.
