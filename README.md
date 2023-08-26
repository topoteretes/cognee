# PromethAI-Memory
Memory management for the AI Applications and AI Agents



![Infographic Image](https://github.com/topoteretes/PromethAI-Memory/blob/main/infographic_final.png)

## The Motivation

Browsing the database of theresanaiforthat.com, we can observe around [7000 new, mostly semi-finished projects](https://theresanaiforthat.com/) in the field of applied AI, whose development is fueled by new improvements in foundation models and open-source community contributions.

It seems it has never been easier to create a startup, build an app, and go to market… and fail.

AI apps currently being pushed out still mostly feel and perform like demos.

To address this issue, [dlthub](https://dlthub.com/) and [prometh.ai](http://prometh.ai/) will collaborate on a productionizing a common use-case, progressing step by step. We will utilize the LLMs, frameworks, and services, refining the code until we attain a clearer understanding of what a modern LLM architecture stack might entail.

### Read more on our blog post [prometh.ai](http://prometh.ai/promethai-memory-blog-post-one)


## PromethAI-Memory Repo Structure

The repository contains a set of folders that represent the steps in the evolution of the modern data stack from POC to production
- Level 1 - CMD script to process PDFs
  We introduce the following concepts:
  1. Structured output with Pydantic
  2. CMD script to process custom PDFs
- Level 2 - Memory Manager implemented in Python

We introduce the following concepts:
  1. Long Term Memory
  2. Short Term Memory
  3. Episodic Buffer
  4. Attention Modulators

The code at this level contains:
  1. Simple PDF ingestion
  2. FastAPI
  3. Docker Image
  4. Memory manager
  5. Langchain-based Agent Simulator
  6. Data schema

## How to use

Each of the folders contains a README to get started. 
