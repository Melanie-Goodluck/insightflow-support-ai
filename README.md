# InsightFlow Support AI

InsightFlow Support AI is a beginner-friendly technical-support RAG project.
It searches fictional product documentation and retrieves the passages most
relevant to a user's question. Later milestones will add embeddings, an LLM,
citations, evaluation, an agent, a web interface, and deployment.

## Milestones 1-6: evaluated RAG and a local web application

Milestone 1 uses a small keyword-based similarity algorithm written with
Python's standard library. Milestone 2 adds a free local sentence-embedding
model so that questions and documents can also be compared by meaning.
Milestone 3 evaluates both retrievers on the same 22 labelled questions using
Top-1 accuracy, Top-3 accuracy, and Mean Reciprocal Rank.
Milestone 4 adds a local FLAN-T5 answer-generation model. The application gives
the model the three semantically retrieved chunks, instructs it to use only
that evidence, and displays the sources separately from the model output.
Milestone 5 evaluates the complete pipeline on 12 answerable and 6
unanswerable questions. It measures answer/refusal routing, expected-source
retrieval, and response latency, then creates a CSV for human review of
correctness, faithfulness, and helpfulness.
Milestone 6 adds a reusable application service, a FastAPI backend, and a
responsive browser interface. The local models are loaded once when the server
starts and reused for every question.

The program:

1. reads the Markdown documents in `data/product_docs`;
2. splits each document into sections called chunks;
3. turns the question and chunks into weighted word representations;
4. compares the question with every chunk;
5. prints the three most relevant chunks and their source files.

## Run it

Python 3.10 or newer is recommended. Create and activate a virtual environment,
then install the semantic-search dependency:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The original lexical application still runs with:

```bash
python app.py
```

Compare lexical and semantic retrieval with:

```bash
python compare_retrievers.py
```

Run the formal evaluation with:

```bash
python evaluate_retrievers.py
```

The command prints aggregate metrics and creates `evaluation_results.csv` for
question-level error analysis.

Run the complete local RAG assistant with:

```bash
python rag_app.py
```

Run the browser application with:

```powershell
python -m uvicorn web_app:app
```

Then open `http://127.0.0.1:8000` in a browser. The API documentation is
available at `http://127.0.0.1:8000/docs`, and the readiness endpoint is
`http://127.0.0.1:8000/api/health`. Avoid development auto-reload on a
memory-constrained computer because it can start an additional Python process.

Run the complete RAG evaluation with:

```bash
python evaluate_rag.py
```

Diagnose one question without rerunning the full dataset with:

```bash
python diagnose_rag.py "How can I invite a colleague to InsightFlow?"
```

The diagnostic command displays the three candidate sections and similarity
scores for the original semantic retriever, a brand-neutral semantic
candidate, and the hybrid experiment. It also shows the raw model output,
grounding coverage, the precise decision reason, and the final answer.
Refusal reasons distinguish low similarity, empty model output, and low
grounding coverage. Candidate evidence is retained even when the final answer
is refused.

The hybrid experiment combines semantic and lexical rankings using reciprocal
rank fusion. It fixed the team-invitation miss, but its 22-question benchmark
was worse than semantic retrieval (72.7% versus 86.4% Top-1 accuracy), so it
was rejected as the main retriever.

The narrower brand-neutral semantic method removes the repeated product name
`InsightFlow` only from the question used to create the query embedding. The
user's original question is preserved for answer generation. It fixed the
observed invitation-query failure while retaining the original semantic
retriever's full benchmark: 86.4% Top-1 accuracy, 100% Top-3 accuracy, and
0.924 Mean Reciprocal Rank. It is therefore used by the normal RAG application
and complete RAG evaluation.

The evaluation creates `rag_evaluation_results.csv`. Automatic metrics report
whether the system answered answerable questions and refused unanswerable
questions. It also records raw model output, refusal reason, grounding score,
answer sources, and candidate evidence. The CSV deliberately leaves three human-review columns blank. Score
correctness, faithfulness, and helpfulness from 0 to 2 after inspecting each
answer against its reference answer and retrieved evidence.

The first run downloads `google/flan-t5-base`. It runs locally on the CPU and
does not make paid generation API calls. A provisional similarity threshold
refuses questions when the best retrieved chunk scores below `0.20`; this
threshold must be evaluated in the next milestone.

FLAN-T5 is loaded explicitly with `AutoTokenizer` and
`AutoModelForSeq2SeqLM`, then generates answers through `model.generate()`.
This avoids relying on version-specific pipeline task aliases and makes the
tokenisation and generation stages explicit.

Two post-generation safeguards handle observed failures. A grounding-coverage
check refuses answers whose meaningful words are poorly supported by retrieved
evidence. A low-information fallback replaces a heading-only or vague short
label with the highest-ranked documentation passage, and only when its
retrieval score is at least `0.35`. Compact factual answers such as comma-separated
format lists and numeric durations remain unchanged. Weaker low-information
matches are refused so that an irrelevant model output cannot trigger an
unrelated full-passage answer. The terminal labels the
top-three chunks as `Retrieved evidence`, since not every candidate necessarily
supports the final answer. These lightweight safeguards are baselines and
require evaluation on answerable and unanswerable questions.

The first semantic run downloads the open-source embedding model and stores it
in the computer's model cache. Later runs reuse the local copy. This project
does not make a paid AI API call.

Try questions such as:

- Why was my CSV file rejected?
- How do I connect a PostgreSQL database?
- Why is my dashboard showing old data?
- What does error E104 mean?

Run the automated checks with:

```bash
python -m unittest discover -s tests -v
```

## Project structure

```text
insightflow-support-ai/
├── data/product_docs/       fictional product knowledge base
├── src/                     retrieval code
├── static/                  browser interface
├── tests/                   automated checks
├── app.py                   command-line entry point
├── web_app.py               FastAPI entry point
└── README.md                project explanation
```

## Current limitation

The current model is intentionally small enough for a 16 GB CPU-based Windows
laptop. The labelled dataset is small and based on fictional documentation, so
its results describe this controlled prototype rather than every real support
knowledge base. Human quality scoring is subjective and should be reported
alongside the scoring rubric and reviewer.
