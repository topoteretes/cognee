# Workflow Migration to Test Suites

This document explains how to ensure all test workflows are only run through the central test-suites.yml workflow.

## Why Migrate to Test Suites?

1. **Prevent Duplicate Runs**: Avoid running the same tests multiple times
2. **Sequential Execution**: Ensure tests run in the correct order
3. **Centralized Control**: Manage all tests from a single place
4. **Resource Efficiency**: Run tests only when needed

## Automated Migration

We've provided a script to automatically convert individual workflows to only run when called by the test-suites.yml file:

```bash
# Make the script executable
chmod +x .github/workflows/disable_independent_workflows.sh

# Run the script
.github/workflows/disable_independent_workflows.sh
```

## Manual Migration

For each workflow file that should only run through test-suites.yml:

1. Open the workflow file
2. Find the `on:` section, which typically looks like:
   ```yaml
   on:
     workflow_dispatch:
     pull_request:
       types: [labeled, synchronize]
   ```

3. Replace it with:
   ```yaml
   on:
     workflow_call:
       secrets:
         inherit: true
   ```

4. Save the file

## Verification

After modifying the workflows, verify that:

1. The workflows no longer trigger on pushes or PRs
2. The workflows still run correctly when called by test-suites.yml
3. No tests are left out of the test-suites.yml orchestrator

## Example Conversion

**Before:**
```yaml
name: test | chromadb

on:
  workflow_dispatch:
  pull_request:
    types: [labeled, synchronize]

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: false

jobs:
  run_chromadb_integration_test:
    name: chromadb test
    runs-on: ubuntu-22.04
    # ...rest of workflow...
```

**After:**
```yaml
name: test | chromadb

on:
  workflow_call:
    secrets:
      inherit: true

jobs:
  run_chromadb_integration_test:
    name: chromadb test
    runs-on: ubuntu-22.04
    # ...rest of workflow...
```

## Special Cases

- **CI/CD Workflows**: Don't modify workflows for CI/CD pipelines like cd.yaml and cd_prd.yaml
- **Shared Workflows**: Keep reusable_*.yml workflows as they are, since they're already designed to be called by other workflows
- **Infrastructure Workflows**: Don't modify workflows that handle infrastructure or deployments
