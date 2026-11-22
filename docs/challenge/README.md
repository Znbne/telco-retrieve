# TelcoRetrieve Challenge
 
In the TelcoRetrieve Challenge you answer telecom questions with a retrieval pipeline and a small language model. The `open-telco-rag challenge` commands turn a question file into a submission file, check it and score it.
 
Submit your file on [Zindi](https://zindi.world/competitions/telcoretrieve-challenge-unlocking-telco-knowledge).
 
The challenge models are fixed: the generator is `google/gemma-3n-E4B-it`, the embedder is OTel-Embedding-0.6B (`otel-0.6b`, hub repo `OTel-Embedding-600M`; fine-tuning allowed) and the reranker is OTel-Reranker-0.6B (`otel-reranker-0.6b`). Other models in the registry are for general use and are not used in the challenge.

Score = 0.7 × accuracy + 0.3 × token efficiency, where token efficiency = 100 × B / (B + T). B is 850 tokens per question and T is the total prompt tokens. Minimum accuracy is 80%. Every `input_prompt` must be at most 4,096 tokens.

The questions are multiple choice or true/false. Answers are scored by exact match, and an empty or invalid answer counts as wrong. B is 850 times the number of questions, and T is the `cl100k_base` token count of `input_prompt` summed over all questions. The minimum accuracy of 80% applies to the whole set. There are 100 public and 200 private questions. Each question may use one generator call, and `input_prompt` is that one prompt.
 
## Rules and clarifications

- **Models.** The challenge models are the embedder OTel-Embedding-0.6B (600M, also called `otel-0.6b`), the reranker OTel-Reranker-0.6B and the generator Gemma 3n E4B-it. You do not have to use all three. For example, Gemma 3n E4B-it with the OTel reranker and BM25, without the embedder, is compliant. Non-model components such as BM25 and heuristics are allowed. Additional learned models are not. The 300M embedder is not used. The pre-built fixed-grid indexes, including `text_baseline/otel-0.6b`, are compliant and need no re-embedding.
- **Fine-tuning.** Fine-tuning is allowed for the generator and/or the embedder, with the same base models, for example an adapter for Gemma 3n E4B-it or a fine-tuned OTel-Embedding-0.6B. OTel-Reranker-0.6B must be used as provided and cannot be fine-tuned. Finalists hand over their fine-tuned weights together with their code in the code review. The reranker used in the benchmark is the OTel one: an OTel reranker and a generic one were compared, and the OTel one performed better and fits the challenge.
- **Training data.** Use only the data provided for the challenge: the GSMA collection on Hugging Face, including the 3GPP documents. Allowed: synthetic data and question-answer pairs generated from the provided documents; open models, including open-weight models run locally, even if they are not one of the three pipeline models; scripts and open-source tools; your own methods for PDF conversion, image and table extraction, hybrid RAG or other pipeline improvements. Tools used only to generate training data do not need to be part of the submitted pipeline, but the generation pipeline may be requested in the code review. The code review uses unseen questions, so training too narrowly on one question style may hurt generalisation.
- **Indexes.** Only the provided Rel-19 documents may be used to build or modify indexes. The pre-built indexes and chunks (`GSMA/telco-retrieve-chunks`) are optional; `telecom-standards-corpus` has the raw documents.
- **AI coding assistants.** They may be used during development only. The submitted pipeline must use only open-source tools and the permitted models, call no paid service or external API, and be fully reproducible from the submitted code.
- **Question formats.** Multiple choice and true/false only, scored by exact match. An abstention counts as wrong. There is no judge model and no citation scoring. Choosing the answer from the first-token (next-token) label probabilities of Gemma 3n E4B-it is allowed if the model is Gemma 3n E4B-it and the prompt is exactly the one reported in `input_prompt`.
- **`input_prompt`.** It must contain all text sent to Gemma 3n E4B-it to produce the submitted answer: the system prompt, any retrieved context, and the question including its answer options as presented to the model. It must work independently: pasted into Gemma 3n E4B-it, it must be enough to produce the submitted answer. The pipeline is limited to one LLM call per question. Omitting text that was actually sent is not permitted. Non-compliant submissions may be disqualified regardless of leaderboard position, and leaderboard positions are provisional.
- **Validation data.** See [Validation questions](#validation-questions).
- **Code sharing.** A private GitHub repository without collaborators is allowed as a backup of your own code. Sharing code or repository access with anyone outside your team is not allowed. Installing the official TelcoRetrieve pipeline from GitHub is allowed and does not count as a custom package.

## 1. Install
 
```bash
git clone https://github.com/Znbne/telco-retrieve
cd telco-retrieve
pip install -e ".[local-embed]"
```
 
For a specific GPU or a CPU-only PyTorch, see [Installation](../reference/installation.md).
 
## 2. Get the questions and the index
 
Download `Test.csv` from the [Zindi challenge page](https://zindi.world/competitions/telcoretrieve-challenge-unlocking-telco-knowledge) (Data tab). It has the columns `ID,question,options` and no answers.
 
The index for the default settings is in [`GSMA/telco-retrieve-chunks`](https://huggingface.co/datasets/GSMA/telco-retrieve-chunks), folder `telcoretrievechallenge/fixed-grid/text_baseline/otel-0.6b` (about 10 GB).
 
```bash
open-telco-rag pull-hf GSMA/telco-retrieve-chunks \
    telcoretrievechallenge/fixed-grid/text_baseline/otel-0.6b --local-dir ./runs
```
 
The database is in `./runs/telcoretrievechallenge/fixed-grid/text_baseline/otel-0.6b/db`. Its collection is `otel-0.6b_text_baseline_b512_v256_full`.
 
To build your own index, see [Build your own index](#build-your-own-index).
 
## 3. Start the generator
 
Serve `google/gemma-3n-E4B-it` with an OpenAI-compatible server such as vLLM. Then set its URL, which ends in `/v1`.
 
```bash
export LOCAL_LLM_BASE_URL=http://localhost:8000/v1
export LOCAL_LLM_API_KEY=...        # only if your server needs one
```
 
`--llm-base-url` does the same as the variable. `--generator-model` (or `LOCAL_LLM_MODEL`) sets the model name sent to the server.
 
## 4. Export, check and score
 
```bash
DB=./runs/telcoretrievechallenge/fixed-grid/text_baseline/otel-0.6b/db
open-telco-rag challenge export --questions Test.csv --out submission.csv \
    --db-path $DB --collection otel-0.6b_text_baseline_b512_v256_full
open-telco-rag challenge check --submission submission.csv --questions Test.csv
open-telco-rag challenge score --submission submission.csv --reference MyAnswers.csv
```
 
`export` writes one row per question in the file you give it.
 
- `export` retrieves passages, builds the prompt, calls the generator and writes the submission. It prints its settings, progress every 25 questions and a last line with the rows, prompt tokens and empty answers.
- `check` validates the file against the questions: header, one row per ID, valid answers, UTF-8 without BOM, LF line endings and the prompt length limit. It exits with status 1 on an error.
- `score` prints accuracy, token efficiency and the overall score against an answer file with the columns `ID,answer`. Use it with your own answers. The leaderboard score is computed by Zindi.
## The submission file
 
The columns are `ID,input_prompt,answer`.
 
- `input_prompt` is the system message, a blank line and the user message, as sent to the generator.
- `answer` is an option letter for multiple choice, or `TRUE` or `FALSE`.
## How tokens are counted
 
A prompt's length is its number of `cl100k_base` tokens (tiktoken). The count covers the whole `input_prompt`: system message, passages, question, options and answer instruction. If a prompt is over `--max-prompt-tokens`, `export` drops the lowest-ranked passages until it fits. Use `--top-k` and `--rerank-top-k` to change how many passages are used.
 
## Default settings
 
These come from [`configs/profiles/challenge.json`](../../configs/profiles/challenge.json). Flags override them.
 
| Setting | Default | Flag |
|---|---|---|
| Embedder | `otel-0.6b` | `--model` |
| Reranker | `otel-reranker-0.6b`, keeps 5 | `--rerank-top-k`, `--no-rerank` |
| Retrieval | hybrid, 10 passages | `--search-method`, `--top-k` |
| Generator | `google/gemma-3n-E4B-it` | `--generator-model` |
| Sampling | temperature 0, seed 1234 | `--temperature`, `--seed` |
 
`open-telco-rag challenge export --help` lists every flag. For your own embedder, prompt builder or answer parser, see [challenge options](../reference/challenge-options.md).
 
## Build your own index
 
```bash
open-telco-rag ingest 3gpp --data DATA --model otel-0.6b --distance-metric cosine \
    --db-path db --collection 3gpp_otel
```
 
`ingest` uses `minilm` and `l2` by default. The published index uses cosine distance, so `--distance-metric cosine` gives your index the same behaviour. `export` works with any metric.
 
## Validation questions

The `GSMA/telco-retrieve-QnA` dataset is not part of the challenge. It was removed from Hugging Face and will not be published, so there is no labelled QnA. Build your own validation questions from the provided Rel-19 documents, using only the permitted models and data. To check your pipeline, save them as JSON and run `open-telco-rag evaluate --questions-file <file>`. See the [evaluation reference](../reference/evaluation.md).

## FAQ
 
**How do I resume a stopped export?** Run the same command with `--resume`. It continues from `<out>.partial`.
 
**`export` says the embedder does not match the index.** Pass the model that built the index, for example `--model otel-0.6b`.
 
**`check` reports an answer that is not an option.** Look at the empty-answer count on the last `export` line, or set `--answer-parser`.
 
**`check` reports a BOM or CR characters.** Write the CSV with LF line endings and no BOM. Do not re-save it in a spreadsheet program.
 
**Rows go missing after I opened the CSV.** IDs such as `123456789012` are text. Do not open or save the file in Excel, which turns them into numbers.
 
