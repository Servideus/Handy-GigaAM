# Code signing policy

Free code signing provided by SignPath.io, certificate by SignPath Foundation

## Roles

- Project owner: Servideus (GitHub user)
- Release managers: Servideus

## Release process

- Release artifacts are built in GitHub Actions from this public repository.
- Windows release artifacts (`.exe` and `.msi`) may be signed with SignPath when SignPath secrets are configured.
- If SignPath secrets are not configured, releases are still published as unsigned.
- Signing credentials and private keys are managed outside this repository and are never committed.

## GitHub Secrets for optional SignPath signing

- `SIGNPATH_API_TOKEN`
- `SIGNPATH_ORGANIZATION_ID`
- `SIGNPATH_PROJECT_SLUG`
- `SIGNPATH_SIGNING_POLICY_SLUG`
- `SIGNPATH_ARTIFACT_CONFIGURATION_SLUG` (optional)
