# AGENTS

## Project Scope

- This workspace is notebook-first and centered on [v4.ipynb](v4.ipynb).
- Topic: binary news classification (Real vs Fake) in Portuguese.
- Methodology: compare traditional NLP/ML pipeline and Transformer-based models.
- Runtime context: Google Colab remote kernel, connected to a university Google account.

## Working Rules For Agents

- Treat this repository as an ipynb editing workflow, not a package/script project.
- Prefer editing existing notebook cells over creating new files, unless the user asks.
- Before creating new functions, cells, or structures, search for equivalent existing logic and reuse/refactor it instead of duplicating code.
- Create new code only when no suitable existing implementation can be reused.
- Keep section structure coherent with the notebook flow (Setup -> Corpus -> EDA -> Cleaning -> Modeling/Evaluation).
- Preserve Portuguese dataset assumptions and labels (`label=0` real, `label=1` fake) unless explicitly changed.

## Environment Assumptions

- Data is expected under `/content/drive/MyDrive/Fake.br-Corpus/`.
- Drive mount is required before data access.
- Colab sessions are ephemeral; avoid assumptions about persisted local state.

## Execution Order (Notebook)

- Run setup and dependency cells before corpus loading or text processing cells.
- Validate path existence before long loops.
- Re-run downstream cells after changing preprocessing functions.

## NLP Conventions For Portuguese

- Preserve Portuguese characters during text normalization when possible.
- Be careful with lemmatization choices: NLTK WordNetLemmatizer is English-focused.
- If proposing preprocessing changes, explain expected impact on Portuguese text quality.

## Validation Checklist Before Finishing

- Confirm required imports and model/package installs are present in notebook cells.
- Confirm paths used in cells match the Drive layout.
- Confirm transformed dataframe columns expected by later cells still exist.
- If execution is requested, report which cells were run and observed outputs/errors.

## Known Pitfalls

- Missing Drive mount causes downstream file/path failures.
- Aggressive regex cleanup may remove Portuguese accents and hurt model quality.
- Partial/out-of-order cell execution can leave stale variables in kernel memory.

## When Unsure

- Ask the user whether to prioritize traditional NLP baseline or Transformer pipeline changes.
- Keep changes minimal and localized to the relevant notebook cells.
