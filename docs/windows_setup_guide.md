# Cognee on Windows — Complete Setup Guide

This guide solves the most common Windows installation issues with Cognee.

## The #1 Problem: Python Version

Cognee requires **Python 3.12** on Windows. Python 3.13+ causes 
C++ binary compilation errors with Cognee's graph database.

### Check your Python version
```bash
python --version
```

If you see 3.13 or higher, follow the steps below.

### Install Python 3.12 on Windows

**Option A: Microsoft Store (Easiest)**
1. Open Microsoft Store
2. Search "Python 3.12"
3. Click Install

**Option B: Official Installer**
1. Go to python.org/downloads/release/python-3129
2. Download "Windows installer (64-bit)"
3. Run installer — CHECK "Add Python to PATH"

### Create a Python 3.12 Virtual Environment
```bash
py -3.12 -m venv venv
venv\Scripts\Activate.ps1
pip install cognee
```

## Required Environment Variables

Create a `.env` file in your project root:
LLM_PROVIDER=gemini
LLM_MODEL=gemini/gemini-2.0-flash
LLM_API_KEY=your_gemini_api_key
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=gemini/gemini-embedding-2
EMBEDDING_API_KEY=your_gemini_api_key
ENABLE_BACKEND_ACCESS_CONTROL=false
COGNEE_SKIP_CONNECTION_TEST=true

**Important:** Set these environment variables BEFORE importing cognee:

```python
import os
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
os.environ["COGNEE_SKIP_CONNECTION_TEST"] = "true"

import cognee
```

## Get a Free Gemini API Key

1. Go to aistudio.google.com
2. Sign in with Google
3. Click "Get API Key"
4. Click "Create API key in new project"
5. Copy the key (starts with AIzaSy...)

## Common Errors and Fixes

### Error: `No suitable Python runtime found`
**Fix:** Install Python 3.12 from Microsoft Store

### Error: `ModuleNotFoundError: No module named 'pydantic_core._pydantic_core'`
**Fix:** Rebuild with Python 3.12:
```bash
Remove-Item -Recurse -Force venv
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install cognee
```

### Error: `An Application Control policy has blocked this file`
**Fix:** Your machine has security restrictions. Use Google Colab instead.

### Error: `429 RESOURCE_EXHAUSTED`
**Fix:** Free Gemini API quota exceeded. Options:
- Create a new Google account for a fresh quota
- Wait for quota reset (midnight Pacific Time)
- Use a different LLM provider

## Verify Installation

```python
import os
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
os.environ["COGNEE_SKIP_CONNECTION_TEST"] = "true"

import asyncio
import cognee

async def test():
    await cognee.remember("Test: Cognee is working on Windows!")
    results = await cognee.recall("Is Cognee working?")
    print("✅ Cognee works!", results)

asyncio.run(test())
```