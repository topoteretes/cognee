# Cognee Local Setup

## Prerequisites

- [Git](https://git-scm.com/)
- [Docker](https://www.docker.com/) / [Colima](https://github.com/abiosoft/colima) (for macOS/Linux container runtime)
- PostgreSQL 14
- OpenAI API Key (or AWS Bedrock credentials if using adapter)

---

## 1. Clone the Repository

```bash
git clone https://github.com/HILabs-Ireland/rules-engine
cd rules-engine
```
## 2. Initialize Submodules

```bash
git submodule update --init
```

## 3. Set Up PostgreSQL

### Install PostgreSQL 14

To install PostgreSQL 14, run:

```bash
brew install postgresql@14
```

### Start PostgreSQL
Start the PostgreSQL service with:

```bash
brew services start postgresql@14
```

### Create the User and Database
Create a PostgreSQL superuser named cognee, set its password, and create the database:

```bash
createuser -s cognee
psql postgres -c "ALTER USER cognee WITH PASSWORD 'cognee'"
createdb -O cognee cognee_db
```

### Install the pgvector Extension
Enable the pgvector extension on the new database:

```bash
psql -d cognee_db -c 'CREATE EXTENSION IF NOT EXISTS vector;'
```

## 4. Obtain OpenAI API Key *(Optional if using Bedrock adapter)*

To use OpenAI models, you'll need an API key. Follow these steps:

1. Go to the [OpenAI Platform](https://platform.openai.com/).
2. Sign up or log in with your email and verify it.
3. Navigate to your [API Keys page](https://platform.openai.com/account/api-keys).
4. Click **+ Create new secret key**.
5. Give it a name (e.g., `cognee`) and copy the key immediately — it won't be shown again.

> ⚠️ The free key includes around $5 of credit. After that, charges will apply.

Keep your key secure — you’ll need it for configuring the `.env` file in the next step.

## 5. Configure Environment Variables

Navigate to the `cognee` directory and copy the example environment file:

```bash
cd cognee
cp .env.template .env
```
Then, open .env in your text editor and update the values as needed. Below is an example configuration:

```env
# Runtime Environment
ENV="local"
DEBUG="true"
TOKENIZERS_PARALLELISM="false"

# Default User Configuration
DEFAULT_USER_EMAIL="default_user@example.com"
DEFAULT_USER_PASSWORD="your_secure_password"

# LLM Configuration
LLM_API_KEY="<your-openai-api-key>"
LLM_MODEL="openai/gpt-4o-mini"
LLM_PROVIDER="openai"
LLM_ENDPOINT=""
LLM_API_VERSION=""
LLM_MAX_TOKENS="16384"

# Embedding Configuration
EMBEDDING_PROVIDER="openai"
EMBEDDING_API_KEY="<your-openai-api-key>"
EMBEDDING_MODEL="openai/text-embedding-3-large"
EMBEDDING_ENDPOINT=""
EMBEDDING_API_VERSION=""
EMBEDDING_DIMENSIONS=3072
EMBEDDING_MAX_TOKENS=8191

# Vector Database Configuration
VECTOR_DB_PROVIDER="pgvector"

# Database Configuration
DB_PROVIDER="postgres"
DB_NAME=cognee_db
DB_HOST=host.docker.internal
DB_PORT=5432
DB_USERNAME=cognee
DB_PASSWORD=cognee
```
💡 Make sure to replace <your-openai-api-key> with your actual key.
If you're using AWS Bedrock, these values may differ depending on your adapter.

## 6. Start the Service

To start the Cognee service locally, run the following command from the project root:

```bash
docker compose up
```

## 7. Testing Cognee with Insomnia

Once the Docker containers are running, you can test the data ingestion and processing workflow using [Insomnia](https://insomnia.rest/).

---

### 1. Authenticate via the Login Endpoint

1. Open Insomnia.
2. Load the relevant Cognee API collection.
3. Find the `Login` endpoint and send a request.
4. A token will be returned and automatically applied to all future requests.

---

### 2. Generate a Pre-Signed S3 URL

1. Go to the **AWS Console → S3 Dashboard**.
2. Locate the bucket: `devrulesenginestack-workflowsta-databuckete3889a50-40uv9d7bnc5e`.
3. Navigate to: `data/Alternaleaf.md`
4. Click **Object Actions** → **Share with presigned URL**.
5. Set the timeout to approximately 5 minutes.
6. Click **Create** — the URL will be copied to your clipboard.
---

### 3. Send the File Link via the Add Data Endpoint

1. In Insomnia, open the **Add Data** endpoint.
2. Paste the pre-signed S3 URL into the request body.
3. Send the request.

> ✅ Expected response: `200 OK` with a `null` body.

---

### 4. Trigger Data Processing

1. Open the **Cognify** endpoint in Insomnia.
2. Send the request.

> ⚠️ This request might time out — that's expected. The processing continues in the background.

---

### 5. Visualize the Results

1. Open the **Visualise** endpoint in Insomnia.
2. Send the request.


## Troubleshooting: PostgreSQL Connection Errors

If you encounter issues connecting to the PostgreSQL database, you may need to reset or reinitialize the database setup.

1. Stop PostgreSQL

```bash
brew services stop postgresql@14
```

2. Remove Existing Data
⚠️ Warning: This will delete all existing PostgreSQL data for version 14.

```bash
rm -rf /opt/homebrew/var/postgresql@14
```

3. Reinitialize PostgreSQL
```bash
initdb /opt/homebrew/var/postgresql@14 -E UTF-8
```

4. Start PostgreSQL Again
```bash
brew services start postgresql@14
```

5. Recreate User and Database
```bash
createuser -s cognee
psql postgres -c "ALTER USER cognee WITH PASSWORD 'cognee'"
createdb -O cognee cognee_db
psql -d cognee_db -c 'CREATE EXTENSION IF NOT EXISTS vector;'
```

