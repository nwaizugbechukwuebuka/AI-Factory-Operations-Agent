# AI Factory Operations Agent

**An AI-powered knowledge assistant for factory operations, technical documentation, and operational decision support.**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688.svg)](https://fastapi.tiangolo.com/)
[![Testing](https://img.shields.io/badge/Testing-pytest-orange.svg)](https://pytest.org/)
[![Code Quality](https://img.shields.io/badge/Code%20Quality-Ruff-purple.svg)](https://docs.astral.sh/ruff/)

## Overview

The AI Factory Operations Agent is a Python-based project designed to make factory knowledge easier to access and use.

Manufacturing environments generate operational documentation covering equipment, safety procedures, troubleshooting, and maintenance. Finding the right information quickly can be difficult when employees must search multiple documents or rely on manual processes.

This project explores how **retrieval-augmented generation (RAG)** can help users retrieve relevant technical information and, as the project develops, obtain AI-generated answers grounded in factory documentation.

The long-term objective is to build a reliable, testable knowledge assistant that supports operational efficiency while reducing unsupported AI-generated answers.

## Business Problem

Factory teams need timely access to accurate operational information.

Common challenges include:

* Searching across multiple technical documents.
* Locating relevant troubleshooting procedures.
* Repeating manual information-retrieval tasks.
* Ensuring answers are supported by approved documentation.
* Evaluating whether an AI knowledge assistant retrieves the correct information.

This project addresses these challenges by combining document retrieval, AI-assisted information processing, and automated evaluation.

## Proposed Solution

The system is being developed in stages:

1. **Knowledge ingestion:** Load factory documents and prepare their content for retrieval.
2. **Semantic search:** Represent document content as embeddings and retrieve relevant information using a vector index.
3. **Retrieval evaluation:** Measure retrieval quality using established information-retrieval metrics.
4. **AI agent:** Generate evidence-based responses and identify supporting sources.
5. **Security and reliability:** Add appropriate safeguards, error handling, and validation.
6. **Deployment readiness:** Prepare the application for demonstration and further development.

The intended outcome is a knowledge assistant that helps users find relevant operational information without treating every generated answer as automatically correct.

## Current Implementation

The project has established a Python application foundation and a document-retrieval evaluation workflow.

| Component                  | Purpose                                                            |
| -------------------------- | ------------------------------------------------------------------ |
| Python                     | Core application and evaluation logic                              |
| FastAPI                    | HTTP API framework                                                 |
| Embedding models           | Convert text into numerical representations for semantic search    |
| FAISS                      | Vector similarity search                                           |
| Document processing        | Prepare technical documents for retrieval                          |
| Recall@k                   | Measure whether relevant documents appear within the top-k results |
| Mean Reciprocal Rank (MRR) | Measure the ranking position of the first relevant result          |
| pytest                     | Automated testing                                                  |
| Ruff                       | Python linting and code-quality checks                             |

**Development status:** The foundation and retrieval-evaluation components are implemented. The test suite and lint checks are still being refined. The evidence-grounded AI agent and its API integration are the next development stage; they should not be considered complete until implemented and tested.

## Technical Architecture

The intended architecture separates document processing, retrieval, answer generation, and API delivery.

```text
             Factory Documentation
                      |
                      v
             Document Processing
                      |
                      v
              Text Chunking
                      |
                      v
             Embedding Generation
                      |
                      v
               FAISS Index
                      |
                      v
              User's Question
                      |
                      v
             Semantic Retrieval
                      |
                      v
             Relevant Evidence
                      |
                      v
             AI Answer Generation
                      |
                      v
            Answer and Source References
```

The diagram represents the target architecture. The answer-generation and source-validation stages remain subject to implementation and testing.

## Retrieval Quality and Evaluation

An AI knowledge assistant must retrieve relevant information before it can provide a useful evidence-based answer.

This project includes evaluation metrics designed to assess the quality of document retrieval.

### Recall@k

Measures whether the expected relevant document appears among the top-k retrieved results.

A higher Recall@k indicates that relevant documents are more consistently included in the retrieved results.

### Mean Reciprocal Rank (MRR)

Measures how highly the first relevant document is ranked.

For a single query:

$$
RR = \frac{1}{\text{rank of the first relevant result}}
$$

MRR is the average reciprocal rank across the evaluation queries.

A higher MRR indicates that relevant documents tend to appear closer to the top of the results.

### Automated Testing

The project uses pytest to validate retrieval metrics and other application behavior.

The evaluation workflow is designed to make retrieval quality measurable rather than relying solely on subjective demonstrations.

Evaluation results will be documented as the test suite stabilizes and the retrieval experiments mature.

## Reliability and Responsible AI

The project is designed around several important engineering principles:

* **Evidence-grounded answers:** Generated answers should be based on retrieved documentation.
* **Source attribution:** Supporting documents should be identified wherever the system can verify their provenance.
* **Graceful handling of missing evidence:** The agent should acknowledge when the available documentation cannot support an answer.
* **Automated evaluation:** Retrieval and application behavior should be tested systematically.
* **Separation of responsibilities:** Retrieval, generation, API handling, and evaluation should remain independently testable.
* **Security by design:** Input validation, access controls, and prompt-injection defenses are planned as part of the security-hardening stage.

These are development objectives, not a claim that every safeguard has already been implemented.

## Technology Stack

| Technology      | Application                                      |
| --------------- | ------------------------------------------------ |
| Python 3.11     | Application development                          |
| FastAPI         | API development                                  |
| FAISS           | Vector similarity search                         |
| Text embeddings | Semantic representation of documents and queries |
| pytest          | Automated testing                                |
| Ruff            | Linting and code quality                         |
| Git and GitHub  | Version control and project collaboration        |

An LLM provider will be integrated as part of the answer-generation stage, subject to the final implementation.

## Running the Project Locally

### Prerequisites

* Python 3.11
* Git
* PowerShell or another terminal
* The project's configured Python dependencies

### 1. Clone the repository

```bash
git clone https://github.com/nwaizugbechukwuebuka/ai-factory-operations-agent.git
cd ai-factory-operations-agent
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If the project defines its development dependencies in `pyproject.toml`, install them using the project's configured installation method.

For example, if the project supports the `dev` optional dependency group:

```powershell
python -m pip install -e ".[dev]"
```

If installation fails, check the dependency configuration in `pyproject.toml` before proceeding.

### 4. Run automated tests

```powershell
python -m pytest -q
```

### 5. Check code quality

```powershell
python -m ruff check .
```

### 6. Start the API

If the FastAPI application is configured at `backend.app.main:app`, run:

```powershell
python -m uvicorn backend.app.main:app --reload
```

The API documentation is typically available at:

`http://127.0.0.1:8000/docs`

The API command and documentation URL assume that the application starts successfully with this module path.

## Project Roadmap

| Phase | Focus                                            | Status                                                 |
| ----- | ------------------------------------------------ | ------------------------------------------------------ |
| 1     | Application foundation and project setup         | Implemented                                            |
| 2     | Document processing, embeddings, and retrieval   | Implemented                                            |
| 3     | Retrieval metrics and automated evaluation       | Implemented; validation and lint issues being resolved |
| 4     | Evidence-grounded AI agent and API integration   | In progress                                            |
| 5     | Security controls, reliability, and auditability | Planned                                                |
| 6     | Deployment, monitoring, and final demonstration  | Planned                                                |

## Skills Demonstrated

This project provides practical experience in:

* Python application development.
* API design with FastAPI.
* Retrieval-augmented generation architecture.
* Semantic search and vector databases.
* Information-retrieval evaluation.
* Automated testing and quality assurance.
* Modular software architecture.
* Responsible AI engineering.
* Technical documentation and iterative development.

## Intended Applications

The architecture could be adapted to support:

* Factory procedure discovery.
* Equipment troubleshooting assistance.
* Technical-documentation search.
* Operational knowledge management.
* Internal support workflows.
* AI-assisted access to approved engineering documentation.

The project is a portfolio prototype, not a certified industrial control system. It is not intended to make autonomous safety-critical decisions or replace approved operating procedures.

## Author

**Tobiloba Nwaizugbe**

Computer Science and Digitisation | AI Applications | Software Quality Assurance | Security Engineering

GitHub: [nwaizugbechukwuebuka](https://github.com/nwaizugbechukwuebuka)