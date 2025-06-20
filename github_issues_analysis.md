# GitHub Issues Analysis for Cognee Contributors

Based on analysis of the codebase and the Slack discussion about making issues public for contributors, here are all the issues, bugs, and TODOs that should be converted to GitHub issues with appropriate labels.

## Categories and Issues

### 🧪 Testing Issues (Label: `testing`, `bug`)

#### High Priority
1. **Regex Entity Extraction Test Failing**
   - **File**: `cognee/tests/unit/entity_extraction/regex_entity_extraction_test.py:85`
   - **Issue**: Test is currently failing and needs to be fixed
   - **TODO Comment**: "TODO: Lazar to fix regex for test, it's failing currently"
   - **Labels**: `bug`, `testing`, `good first issue`

2. **Neo4j Test Disabled Due to LLM Model Issues**
   - **File**: `cognee/tests/test_neo4j.py:78`
   - **Issue**: Test fails often on weak LLM models and has been removed
   - **Labels**: `bug`, `testing`, `llm-related`

3. **RAG Completion Retriever top_k Parameter Bug**
   - **File**: `cognee/tests/unit/modules/retrieval/rag_completion_retriever_test.py:155`
   - **Issue**: top_k doesn't affect the output, needs to be fixed
   - **Labels**: `bug`, `retrieval`, `testing`

### 🏗️ Infrastructure & CI/CD (Label: `infrastructure`, `contributor-experience`)

#### High Priority
1. **Contributor PR Testing Workflow Automation**
   - **Context**: From Slack thread - need automated CI/CD for contributor PRs
   - **Current**: Manual process requiring branch creation and manual test runs
   - **Goal**: Automate the workflow described in Notion page
   - **Labels**: `infrastructure`, `contributor-experience`, `enhancement`

2. **GitHub Issue Management System**
   - **Context**: From Slack thread - need systematic approach to creating and managing GitHub issues
   - **Goal**: Template system and regular process for adding new issues
   - **Labels**: `meta`, `contributor-experience`, `good first issue`

### 🔧 Technical Debt & TODOs (Label: `technical-debt`, `enhancement`)

#### Medium Priority
1. **User Role Permissions System**
   - **File**: `cognee/modules/users/permissions/methods/check_permission_on_documents.py:19`
   - **Issue**: User role permissions temporarily disabled during rework
   - **Labels**: `enhancement`, `permissions`, `technical-debt`

2. **Default Tasks Configuration**
   - **File**: `cognee/api/v1/cognify/cognify.py:41`
   - **Issue**: Need better way to handle default tasks configuration
   - **Comment**: "Boris's comment" - find better solution
   - **Labels**: `enhancement`, `api`, `technical-debt`

3. **Graph Database ID System**
   - **File**: `cognee/modules/data/models/GraphMetrics.py:12`
   - **Issue**: Change ID to reflect unique id of graph database
   - **Labels**: `enhancement`, `database`, `technical-debt`

4. **Pipeline Task Assignment**
   - **File**: `cognee/modules/pipelines/operations/pipeline.py:145`
   - **Issue**: Random assignment needs permanent solution
   - **Labels**: `enhancement`, `pipeline`, `technical-debt`

5. **Datasets Permission Handling**
   - **File**: `cognee/api/v1/datasets/routers/get_datasets_router.py:85`
   - **Issue**: Handle situation differently if user doesn't have permission to access data
   - **Labels**: `enhancement`, `permissions`, `api`

6. **Model Support Extension**
   - **File**: `cognee/api/v1/responses/routers/get_responses_router.py:53`
   - **Issue**: Support other models (e.g. cognee-v1-openai-gpt-3.5-turbo, etc.)
   - **Labels**: `enhancement`, `llm-support`, `api`

### 🐛 Bug Fixes (Label: `bug`)

#### Medium Priority
1. **Pipeline Status Update Lock**
   - **File**: `cognee/modules/pipelines/operations/pipeline.py:139`
   - **Issue**: UI lock needed to prevent multiple backend requests
   - **Current**: Commented out with TODO
   - **Labels**: `bug`, `ui`, `concurrency`

2. **Network Visualization Properties**
   - **File**: `cognee/modules/visualization/cognee_network_visualization.py:36`
   - **Issue**: Decide what properties to show on nodes and edges
   - **Labels**: `enhancement`, `visualization`, `ux`

### 📚 Documentation & User Experience (Label: `documentation`, `good first issue`)

#### Low Priority  
1. **Code Processing Documentation**
   - **File**: Multiple files with "Notes:" sections need better documentation
   - **Files**: 
     - `cognee/tasks/documents/extract_chunks_from_documents.py:34`
     - `cognee/tasks/graph/extract_graph_from_code.py:14`
     - `cognee/tasks/documents/check_permissions_on_documents.py:11`
   - **Labels**: `documentation`, `good first issue`

2. **Test Input Documentation**
   - **File**: `cognee/tests/unit/processing/chunks/test_input.py:141`
   - **Issue**: Document that keyword arguments are exclusive to one or more contexts
   - **Labels**: `documentation`, `testing`, `good first issue`

### 🔧 Minor Code Quality Issues (Label: `code-quality`, `good first issue`)

1. **Remove Ugly Hack**
   - **File**: `cognee/modules/pipelines/operations/pipeline.py:114`
   - **Issue**: Replace "ugly hack" with proper solution
   - **Labels**: `code-quality`, `refactor`, `good first issue`

2. **Extract Chunks TODO**
   - **File**: `cognee/tasks/documents/extract_chunks_from_documents.py:46`
   - **Issue**: Incomplete TODO comment "todo rita"
   - **Labels**: `good first issue`, `clarification-needed`

## Recommended GitHub Issue Labels

### Priority Labels
- `priority-high` - Critical issues affecting functionality
- `priority-medium` - Important improvements
- `priority-low` - Nice-to-have enhancements

### Type Labels  
- `bug` - Something isn't working
- `enhancement` - New feature or improvement
- `technical-debt` - Code quality improvements
- `documentation` - Documentation improvements
- `testing` - Test-related issues

### Area Labels
- `api` - API-related issues
- `database` - Database-related issues  
- `llm-support` - LLM integration issues
- `ui` - User interface issues
- `infrastructure` - CI/CD and development workflow
- `permissions` - Authentication/authorization
- `retrieval` - Information retrieval functionality
- `visualization` - Data visualization features

### Contributor Labels
- `good first issue` - Perfect for newcomers
- `contributor-experience` - Issues affecting contributor workflow
- `help wanted` - Extra attention needed

## Next Steps

1. **Create Issue Templates**: Set up templates for bug reports, feature requests, and enhancements
2. **Batch Issue Creation**: Create all identified issues with proper titles, descriptions, and labels
3. **Contributor Onboarding**: Update CONTRIBUTING.md with information about these issues
4. **Regular Review Process**: Establish weekly review to add new issues as discussed in Slack

## Issue Creation Priority Order

1. **High Priority Testing Issues** - These affect functionality and should be fixed first
2. **Infrastructure Issues** - Will improve contributor experience immediately  
3. **Medium Priority TODOs** - Technical debt that affects maintainability
4. **Documentation & Good First Issues** - Perfect for new contributors to get started

This analysis found **15+ distinct issues** that should be converted to GitHub issues, ranging from critical bugs to good first issues for new contributors.