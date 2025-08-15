# Cross-Scale Interaction CSVs

This repo contains CSV outputs from model runs that detect **cross-scale interactions** in text. The included script runs Claude, OpenAI, and Gemini with different prompts and writes a row per paragraph that any model flags as a valid interaction.

## Folder overview

- **Mixed Prompt/**
  - CSVs produced with the **comprehensive** prompt that includes *all defined scales* across tasks  
    (e.g. temporal: Short/Medium/Long).

- **Old Definitions/**
  - CSVs produced with the **earlier set from Ju Young** that treats the **three scales as separate individual scales**.  

- **New Definitions/**
  - CSVs produced with the **stricter revision** of the definitions.  
  - Use these for the highest bar on “explicit mechanism” and tighter acceptance criteria.
  - You can read through the prompts as I've attached the python files 

> In short: **Mixed Prompt** = most comprehensive; **Old Definitions** = earlier, looser three-scale set; **New Definitions** = stricter version.
> In the propmts you can read about the pattern types and how the direction is being represented

## What each CSV contains

Each CSV is filtered to paragraphs where **at least one model** said an interaction is present.

Common columns:

- `Source` — Document identifier or path for the paragraph.
- `Full Paragraph Text` — The raw paragraph analyzed.
- `Temporal Interaction Found` — “Yes” if any model flagged it.
- `Disagreement` — “Yes” if models disagreed on presence.

Per-model columns appear **once per model tag** in `MODELS`:

Model tags:
- `claude3.7` → Claude 3.7 Sonnet  
- `4o` → OpenAI GPT-4o  
- `o4_mini` → OpenAI GPT-4o-mini  
- `gemini2.5` → Gemini 2.5 Flash

For each model tag:

- `<tag>: Interaction` — “Yes” or “No”
- `<tag>: Confidence` — 0.0–1.0 if returned
- `<tag>: Timeframes` — Detected scales (e.g., “Short-term, Long-term”) or “N/A”
- `<tag>: Explanations` — Brief justifications per detected scale or “N/A”
- `<tag>: Direction` — Direction label if provided, otherwise “N/A”
- `<tag>: Type` — Relationship type  
  `Direct Causal Link | Causal Chain | Convergent Influence | Bidirectional Link | Feedback Loop | Mixed/Complex Pattern | N/A`
- `<tag>: Summary` — One-sentence summary with a bracketed prefix, or “N/A”

## How these were generated

The script:

1. Loads paragraphs from `all_nestle_unilever_paragraphs_fixed.json`
2. Sends each paragraph to prompts and models:
   - **Temporal prompts** define Short/Medium/Long with explicit mechanism requirement.
   - **Spatial prompt** defines Local/Regional/National/Transnational with the same rule.
   - **Hierarchical propmts** defines Individual/Organizational/Environmental.....
3. Parses returned JSON and writes one combined CSV per run.
4. Keeps a paragraph **only** if at least one model says “Yes”.

Example command:

```bash
python detect_cross_scale.py \
  -i paragraphs_comparison.json \
  -o temporal_results.csv \
  -n 50
