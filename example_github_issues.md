# Example GitHub Issues

These are examples of how to create GitHub issues for the identified problems using the templates.

## Example 1: High Priority Bug

**Title**: [BUG] Regex Entity Extraction Test Failing

**Labels**: `bug`, `testing`, `good first issue`

**Content**:
```markdown
## Bug Description
The regex entity extraction test is currently failing and needs to be fixed. This test is part of the entity extraction module and prevents proper validation of the regex functionality.

## Location
**File(s)**: `cognee/tests/unit/entity_extraction/regex_entity_extraction_test.py`
**Line(s)**: 85

## Current Behavior
The test is failing and has been marked with a TODO comment indicating it needs to be fixed.

## Expected Behavior
The regex entity extraction test should pass and properly validate the regex entity extraction functionality.

## Steps to Reproduce
1. Run the test suite
2. Execute the specific test: `python -m pytest cognee/tests/unit/entity_extraction/regex_entity_extraction_test.py`
3. Observe test failure

## Environment
- All environments affected
- Python Version: 3.8+
- Cognee Version: Current dev branch

## Error Output
```
# Test failure output would go here
```

## Additional Context
TODO comment from line 85: "TODO: Lazar to fix regex for test, it's failing currently"

## Related Issues
Part of the broader testing infrastructure improvements needed for contributor PRs.

## Priority
- [x] High - Blocks functionality
- [ ] Medium - Affects user experience  
- [ ] Low - Minor issue

## Labels to Add
- [x] `testing` - Related to tests
- [x] `good first issue` - Good for newcomers
```

## Example 2: Infrastructure Enhancement

**Title**: [ENHANCEMENT] Automate Contributor PR Testing Workflow

**Labels**: `enhancement`, `infrastructure`, `contributor-experience`

**Content**:
```markdown
## Enhancement Description
Automate the contributor PR testing workflow to eliminate manual branch creation and testing processes.

## Problem/Motivation
Currently, testing contributor PRs requires manual intervention:
1. Create a new branch from contributor's PR
2. Push to origin manually
3. Run test_suites from GitHub pointing to the branch

This creates friction for maintainers and delays feedback to contributors.

## Location (if applicable)
**File(s)**: `.github/workflows/` (various workflow files)
**TODO/Comment Reference**: From Slack discussion about contributor PR workflow automation

## Proposed Solution
Implement automated workflow that:
1. Automatically creates test branches for contributor PRs
2. Runs full test suite on contributor PRs
3. Provides security scanning for malicious GitHub Action changes
4. Reports results back to the PR

## Alternative Solutions
- Use GitHub's built-in PR testing with security restrictions
- Fork-based testing with approval gates
- Separate testing environment for external contributions

## Implementation Details
- [x] API changes required
- [ ] Database schema changes
- [ ] Breaking changes
- [ ] New dependencies needed
- [x] Documentation updates needed

## Acceptance Criteria
- [ ] Contributor PRs automatically trigger test suite
- [ ] Security scanning prevents malicious workflow changes
- [ ] Test results are reported back to PR
- [ ] Maintainer approval still required for merge
- [ ] Documentation updated with new workflow

## Additional Context
Current workflow documented in Notion page referenced in Slack discussion. Need to build upon the existing improved workflow.

## Related Issues/TODOs
Addresses the manual contributor PR testing process discussed in engineering Slack channel.

## Priority
- [x] High - Critical for functionality
- [ ] Medium - Important improvement
- [ ] Low - Nice-to-have

## Labels to Add
- [x] `infrastructure` - CI/CD and development workflow
- [x] `contributor-experience` - Issues affecting contributor workflow
- [x] `help wanted` - Extra attention needed
```

## Example 3: Good First Issue

**Title**: [GOOD FIRST ISSUE] Remove "Ugly Hack" in Pipeline Operations

**Labels**: `good first issue`, `code-quality`, `refactor`

**Content**:
```markdown
## Issue Description
Replace a self-documented "ugly hack" in the pipeline operations with a proper solution.

## Location
**File(s)**: `cognee/modules/pipelines/operations/pipeline.py`
**Line(s)**: 114
**Function/Class**: Pipeline operation handling

## What Needs to Be Done
1. Review the current code at line 114 to understand what the "hack" is doing
2. Research the proper way to implement this functionality
3. Replace the hack with clean, maintainable code
4. Ensure all tests still pass
5. Update any related documentation

## Context/Background
The current code contains a comment "# Ugly hack, but no easier way to do this." This indicates technical debt that should be addressed. Clean code is important for maintainability and contributor onboarding.

## Expected Outcome
The functionality should work exactly the same, but with clean, well-documented code instead of a hack.

## Getting Started
### Setup Instructions
1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/cognee.git`
3. Install dependencies: `uv sync --all-extras` or `pip install -e .`
4. Create a branch: `git checkout -b fix/remove-pipeline-hack`

### Testing
How to test the changes:
```bash
# Run pipeline-related tests
python -m pytest cognee/tests/ -k pipeline
# Run full test suite to ensure no regressions
python -m pytest
```

### Files to Focus On
- Primary file to modify: `cognee/modules/pipelines/operations/pipeline.py`
- Related files to review: Other files in `cognee/modules/pipelines/`
- Test files to update: `cognee/tests/` (pipeline-related tests)

## Acceptance Criteria
- [ ] "Ugly hack" comment removed
- [ ] Functionality replaced with clean implementation
- [ ] All existing tests pass
- [ ] Code follows project style guidelines
- [ ] Added comments explaining the new implementation

## Resources
- [Contributing Guide](../../CONTRIBUTING.md)
- [Code of Conduct](../../CODE_OF_CONDUCT.md)
- Python best practices documentation

## Need Help?
- Join our [Discord community](https://discord.gg/bcy8xFAtfd)
- Comment on this issue with questions
- Tag maintainer for guidance if stuck

## Estimated Time
⏱️ **Estimated time**: 1-2 hours

## Skills Needed
- [x] Python basics
- [x] Git/GitHub
- [ ] Testing (optional)
- [ ] Documentation (optional)

## Additional Context
This is a great first issue because it's well-contained, has clear success criteria, and helps improve code quality. The exact nature of the "hack" will become clear when reviewing the code.
```

## Quick Issue Creation Checklist

When creating issues from the analysis document:

### For Each Issue:
1. **Choose appropriate template** (Bug Report, Enhancement, Documentation, or Good First Issue)
2. **Add proper labels** from the recommended label list
3. **Set priority level** (High, Medium, Low)
4. **Reference the source** (file and line number from analysis)
5. **Provide context** from the analysis document
6. **Make it actionable** with clear acceptance criteria

### Batch Creation Process:
1. Start with **High Priority Testing Issues** (3 issues)
2. Create **Infrastructure Issues** (2 issues)  
3. Add **Technical Debt Issues** (6 issues)
4. Create **Documentation Issues** (2 issues)
5. Add **Code Quality Issues** (2 issues)

### Total Issues to Create: 15+

This systematic approach will provide contributors with a clear path to contribute, from simple documentation fixes to more complex infrastructure improvements.