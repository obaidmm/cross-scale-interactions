#!/usr/bin/env python3
"""
Detect hierarchical cross-scale interactions in paragraphs with Claude, OpenAI, and Gemini models.
then write the structured results to a CSV.
"""
import config
import argparse
import json
import re
import time
from pathlib import Path


import pandas as pd
import anthropic
from openai import OpenAI
from google import genai

# ---------------------------------------------------------------------------
# API setup
# ---------------------------------------------------------------------------
claude        = anthropic.Client(api_key=config.ANTHROPIC_API_KEY)
client        = OpenAI(api_key=config.OPENAI_API_KEY)
client_gemini = genai.Client(api_key=config.GOOGLE_API_KEY)


MODELS = {
    "claude3.7":    "claude-3-7-sonnet-20250219",
    "4o":           "gpt-4o",
    "o4_mini":      "gpt-4o-mini",
    "gemini2.5":    "gemini-2.5-flash"
}

# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------

CLAUDE_DETECTOR_PROMPT = r"""
<task>
You are a meticulous research analyst operating under a strict protocol to identify and classify **hierarchical cross-scale interactions** in text.
Your function is to determine if an entity at one defined scale *explicitly and causally* affects, or is affected by, an entity at a different defined scale. Adherence to the protocol is mandatory.
</task>

<analytical_protocol>
You must follow these steps in order:
1.  **Identify Actors & Actions**: First, identify all active entities (the actors) and the specific actions they are taking in the provided text.
2.  **Map to Scales**: Map each identified actor to one of the official scales defined below. If an actor cannot be mapped, it cannot be part of an interaction.
3.  **Verify Causal Link**: For each action, determine if it creates a clear and non-ambiguous causal link between two or more different scales. The text must describe the influence; do not infer it.
4.  **Classify Pattern**: If a cross-scale causal link exists, classify its structure using the defined causal patterns.
5.  **Generate Output**: Construct the final JSON output according to the strict rules and format.
</analytical_protocol>

<scales>
  <scale name="Individual">
    <description>Specific people or informal groups (e.g., workers, customers, households).</description>
    <examples>A worker's job dissatisfaction; An individual consumer’s boycott.</examples>
  </scale>
  <scale name="Organizational">
    <description>A single company, NGO, agency, or formal group (e.g., a brand or internal team), even if cross-border.</description>
    <examples>A company’s ESG policy; A hospital running a waste audit.</examples>
    <clarification>'Community' belongs under 'Environmental and Societal', not here. The UN as one body is Organizational.</clarification>
  </scale>
  <scale name="Inter-Organizational / Sector">
    <description>Collaborations or systemic links between multiple organizations (e.g., coalitions, supply chains).</description>
    <examples>Industry alliances; the global supply chain.</examples>
    <clarification>Distinguish from one organization. 'The supply chain' = Inter-Org; 'one supplier' = Organizational.</clarification>
  </scale>
  <scale name="Environmental and Societal">
    <description>Macro-level structures like governments, ecosystems, or societies.</description>
    <examples>Capital markets; community norms; forest land use limitations.</examples>
    <clarification>'Community' as a general group or force fits here.</clarification>
  </scale>
</scales>

<causal_patterns>
  <pattern name="Direct Causal Link">A -> B. A direct, unidirectional influence across two different scales.</pattern>
  <pattern name="Causal Chain">A -> B -> C. A sequential influence across three or more distinct scales.</pattern>
  <pattern name="Convergent Influence">[A, B] -> C. Multiple distinct scales independently influence a common target scale.</pattern>
  <pattern name="Bidirectional Link">A <-> B. Two scales mutually influence each other.</pattern>
  <pattern name="Feedback Loop">A -> B -> A. A cyclical influence where effects return to the origin scale.</pattern>
  <pattern name="Mixed/Complex Pattern">A hybrid of the above patterns, including loops or forked pathways.</pattern>
</causal_patterns>

<strict_rules>
  - **Actor-Driven Causality Only**: Your analysis MUST be based on explicit actions taken by identified actors. Do not infer interactions from abstract topics, correlations, or vague associations.
  - **Explicit Link Required**: A hierarchical interaction is present ONLY if the text describes a clear causal mechanism between at least two different scales. If the link is merely plausible but not stated, the interaction is not present.
  - **Adherence to Definitions**: You MUST use the exact scale and pattern names provided. Do not invent or modify them.
  - **Pathway Formatting**: All causal pathways MUST be written using arrows (`->`, `<->`) and include the official named scales in the correct causal order.
  - **Output Integrity**: The final output MUST be a single, valid JSON object and nothing else.
</strict_rules>

<output_instructions>
  1. "hierarchical_interaction_present": "Yes" or "No".
  2. "confidence": A float from 0.0 to 1.0. Use 1.0 for explicitly stated causality, <0.8 for strongly implied causality.
  3. "scales_reasoning":
     - "detected_scales": List all distinct scales involved.
     - "scale_explanations": Brief justification for mapping each actor to its scale.
  4. "causal_pattern":
     - "type": One of the listed causal pattern names.
     - "pathway": Use `->` or `<->` to show causal flow. Include all relevant scales in order (e.g., "Individual -> Organizational").
  5. "summary": A one-sentence description (max 200 characters) starting with the bracketed causal pathway, e.g., "[Individual -> Organizational] An employee's whistleblowing forces corporate change."

  - If "hierarchical_interaction_present" is "No", all other fields MUST be set to "N/A" or be empty.
</output_instructions>

<output_format>
{
  "hierarchical_interaction_present": "Yes|No",
  "confidence": 0.0-1.0,
  "scales_reasoning": {
    "detected_scales": ["Scale A", "Scale B"],
    "scale_explanations": {
      "Scale A": "Example explanation.",
      "Scale B": "Example explanation."
    }
  },
  "causal_pattern": {
    "type": "Direct Causal Link",
    "pathway": "Individual -> Organizational"
  },
  "summary": "[Individual -> Organizational] ..."
}
</output_format>
"""


OPENAI_DETECTOR_PROMPT = r"""
/*
TASK
You are a precision analyst operating under a strict protocol to detect and classify **hierarchical cross-scale interactions**.
A hierarchical interaction occurs ONLY when an entity at one scale explicitly and causally influences, or is influenced by, an entity at another scale. You must adhere to the protocol below without deviation.

────────────────────────────────────
ANALYTICAL PROTOCOL (Follow these steps internally before responding)
1.  IDENTIFY ACTORS & ACTIONS: Scan the text to identify all actors and the specific actions they perform.
2.  MAP SCALES: Assign each actor to one of the official HIERARCHICAL SCALE DEFINITIONS below. An actor that cannot be mapped cannot participate in an interaction.
3.  VERIFY CAUSALITY: For each action, determine if the text explicitly describes a causal link between two or more different scales. Correlation is not causation. If the link is not described, it does not exist.
4.  CLASSIFY PATTERN: If a valid cross-scale interaction is found, classify its structure using the CAUSAL PATTERN DEFINITIONS.
5.  GENERATE JSON: Construct the final JSON output according to the STRICT OUTPUT DIRECTIVES.

────────────────────────────────────
HIERARCHICAL SCALE DEFINITIONS
• Individual: Specific individuals or informal groups (workers, customers, a family).
• Organizational: A single organization (company, NGO, agency, brand), its internal units, or formal groups (worker associations). Note: The UN as a body is Organizational.
• Inter-Organizational / Sector: Groups of organizations (supply chains, industry alliances). Note: 'The supply chain' (Inter-Org) vs. a specific 'supplier' (Organizational).
• Environmental and Societal: Societal-level systems (governments, capital markets, communities, natural systems).

────────────────────────────────────
CAUSAL PATTERN DEFINITIONS
• Direct Causal Link: A -> B. An entity at scale A directly causes an effect in an entity at scale B.
• Causal Chain: A -> B -> C. A sequence where A influences B, which then influences C.
• Convergent Influence: [A, B] -> C. Multiple scales independently influence a common target scale.
• Bidirectional Link: A <-> B. Two scales mutually influence each other.
• Feedback Loop: A -> B -> A. A closed loop where effects cycle back to the origin.
• Mixed/Complex Pattern: A hybrid or non-linear combination of other patterns.

────────────────────────────────────
STRICT OUTPUT DIRECTIVES
- YOU MUST focus on the actors and their specified activities. Do not analyze abstract topics.
- YOU WILL NOT infer causality. An interaction requires a link to be explicitly described in the text between at least two different scales.
- IF no interaction is present, you MUST flag "No" and leave other fields as "N/A".
- YOUR FINAL OUTPUT must be a single, raw JSON object. Do not use Markdown code fences or add any explanatory text outside the JSON structure.
- YOU MUST use the exact scale and pattern names as provided.

────────────────────────────────────
WHAT TO RETURN (pure JSON, no code fences):
1. hierarchical_interaction_present: "Yes" or "No"
2. confidence: 0.0 - 1.0 (Use 1.0 for explicitly stated causality, <0.8 for strongly implied causality).
3. scales_reasoning:
   - detected_scales: List of scales found (e.g., ["Individual", "Organizational"]).
   - scale_explanations: A mapping of scale -> brief justification for the mapping.
4. causal_pattern:
   - type: The name of the causal pattern identified (e.g., "Direct Causal Link").
   - pathway: The string showing the causal flow (e.g., "Individual -> Organizational").
5. summary:
   - One concise sentence (max 200 characters) starting with the bracketed pathway from the causal_pattern.
   - Example: "[Individual -> Organizational] An employee's whistleblowing prompts a new corporate policy."

────────────────────────────────────
OUTPUT FORMAT
{
  "hierarchical_interaction_present": "Yes|No",
  "confidence": 0.0-1.0,
  "scales_reasoning": {
    "detected_scales": ["Scale A", "Scale B"],
    "scale_explanations": {
      "Scale A": "…",
      "Scale B":  "…"
    }
  },
  "causal_pattern": {
    "type": "Causal Chain",
    "pathway": "Individual -> Organizational -> Environmental and Societal"
  },
  "summary": "[Individual -> Organizational -> Environmental and Societal] …"
}
*/
"""

GEMINI_DETECTOR_PROMPT = r"""
TASK:
You are a specialist research system operating under a strict protocol to detect and classify **hierarchical cross-scale interactions**.
A hierarchical interaction exists ONLY when an entity at one defined scale explicitly and causally influences, or is influenced by, an entity at another defined scale. You must follow the protocol without deviation.

---
**ANALYTICAL PROTOCOL**
You must execute the following reasoning steps before generating your output:
1.  **Actor Identification**: Isolate the specific actors in the text and the actions they perform.
2.  **Scale Mapping**: Assign each actor to one of the official scales defined below.
3.  **Causality Verification**: Confirm that the text describes a direct causal link between actions and outcomes across at least two different scales. Do not infer causality from correlation.
4.  **Pattern Classification**: If a link is verified, classify its structure using the defined causal patterns.
5.  **JSON Generation**: Construct the output according to the strict directives.

---
**HIERARCHICAL SCALE DEFINITIONS**

* **Individual**: Specific individuals or informal groups (workers, employees, customers).
* **Organizational**: A single organization (company, NGO, brand), its internal units, or formal groups. *Clarification*: The UN as a single body is Organizational.
* **Inter-Organizational / Sector**: Groups of organizations (supply chains, industry alliances). *Clarification*: 'The supply chain' as a system (Inter-Org) vs. a specific 'supplier' (Organizational).
* **Environmental and Societal**: Societal-level systems (governments, capital markets, communities, natural systems).

---
**CAUSAL PATTERN DEFINITIONS**

* **Direct Causal Link**: `A -> B`. A direct, one-way influence between two scales.
* **Causal Chain**: `A -> B -> C`. A sequential influence across three or more entities/scales.
* **Convergent Influence**: `[A, B] -> C`. Multiple scales affect a single target scale.
* **Bidirectional Link**: `A <-> B`. Mutual, two-way influence between two scales.
* **Feedback Loop**: `A -> B -> A`. A circular causal pattern where effects return to the origin.
* **Mixed/Complex Pattern**: A hybrid or non-linear combination of the above.

---
**STRICT OUTPUT DIRECTIVES**

1.  **Focus on Actors**: Your analysis must be based on an activity tied to an actor, not a general topic.
2.  **Interaction Required**: An interaction requires a non-ambiguous, described causal link between at least two different scales.
3.  **Adhere to Definitions**: You must use the exact scale and pattern names provided. No modifications are permitted.
4.  **Output Format**: You must respond with a single, valid JSON object only. Do not include any text, explanations, or Markdown code fences before or after the JSON object.
5.  **Empty Fields**: If `hierarchical_interaction_present` is "No", all other fields must be present but set to "N/A" or an empty equivalent.

---
**OUTPUT INSTRUCTIONS**

1.  `hierarchical_interaction_present`: "Yes" or "No".
2.  `confidence`: Float from 0.0 to 1.0. (1.0 for explicitly stated causality, <0.8 for strongly implied causality).
3.  `scales_reasoning`:
    * `detected_scales`: A list of the scales you found.
    * `scale_explanations`: A dictionary explaining why the text fits each identified scale.
4.  `causal_pattern`: An object containing:
    * `type`: The name of the pattern from the definitions (e.g., "Feedback Loop").
    * `pathway`: The string representing the causal flow (e.g., "Organizational -> Environmental and Societal -> Organizational").
5.  `summary`: A single sentence (max 200 characters) summarizing the interaction, starting with the bracketed pathway. Example: `"[Individual -> Organizational] An employee's whistleblowing forces a change in corporate policy."`

---
**OUTPUT FORMAT**
```json
{
  "hierarchical_interaction_present": "Yes|No",
  "confidence": 0.0-1.0,
  "scales_reasoning": {
    "detected_scales": ["Scale A", "Scale B", "Scale C"],
    "scale_explanations": {
      "Scale A": "Example explanation.",
      "Scale B": "Example explanation.",
      "Scale C": "Example explanation."
    }
  },
  "causal_pattern": {
    "type": "Causal Chain",
    "pathway": "Scale A -> Scale B"
  },
  "summary": "[Scale A -> Scale B] ..."
}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def analyze_claude(model_id: str, paragraph: str) -> str:
    """Call Claude with the temporal prompt."""
    resp = claude.messages.create(
        model=model_id,
        system=CLAUDE_DETECTOR_PROMPT,
        messages=[{"role": "user", "content": paragraph.strip()}],
        max_tokens=600,
        temperature=0
    )
    return resp.content[0].text


def analyze_openai(model, text):
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        # max_completion_tokens=600,
        max_tokens=600,
        response_format={"type": "json_object"},
        messages=[
            {"role":"system", "content":OPENAI_DETECTOR_PROMPT},
            {"role":"user",   "content":text.strip()}
        ]
    )
    return resp.choices[0].message.content

def analyze_gemini(model, paragraph):
    """Call Gemini with the hierarchical prompt."""
    # The 'context' variable was not used, so we simplify the input
    full_prompt = GEMINI_DETECTOR_PROMPT + "\n" + paragraph.strip()
    gen_config = {}

    res = client_gemini.models.generate_content(
        model=f"models/{model}", 
        contents=[full_prompt],
        **gen_config
    )

    return res.text


def parse_json(block: str):
    """Robustly pull the first JSON object from a model reply."""
    if not isinstance(block, str):
        return None
    match = re.search(r'```(?:json)?\s*({.*?})\s*```', block, re.DOTALL)
    if not match:
        start, end = block.find('{'), block.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return None
        json_str = block[start:end + 1]
    else:
        json_str = match.group(1)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # quick clean-ups for minor formatting slips
        try:
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', json_str)
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Detect hierarchical interactions in text using various LLMs.")
    parser.add_argument("-i", "--input", default="paragraphs_comparison.json", help="Input JSON file with paragraphs.")
    parser.add_argument("-o", "--output", default="hierarchical_results.csv", help="Output CSV file for results.")
    parser.add_argument("-n", "--limit", type=int, default=None, help="Limit the number of paragraphs to process.")
    args = parser.parse_args()

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            paragraphs_data = json.load(f)
    except Exception as e:
        print(f"Failed to load input: {e}")
        return

    paragraphs = paragraphs_data[:args.limit] if args.limit else paragraphs_data
    if not paragraphs:
        print("No data found.")
        return

    results = []
    total = len(paragraphs)

    print(f"\nStarting hierarchical interaction scan on {total} paragraph(s)...\n")

    for idx, p_data in enumerate(paragraphs):
        text = p_data["paragraph"]
        source = p_data["pdf"]
        rec = {"Source": source, "Full Paragraph Text": text}
        model_status = {}
        model_scales = {}

        print(f"\n--- Paragraph {idx + 1} of {total} ---")
        print(f"Source: {source}")

        for tag, mid in MODELS.items():
            try:
                print(f"[{idx + 1}/{total}] Querying {tag}...", end=" ", flush=True)
                t0 = time.time()

                if tag.startswith("claude"):
                    response = analyze_claude(mid, text)
                elif tag.startswith("gemini"):
                    response = analyze_gemini(mid, text)
                else:
                    response = analyze_openai(mid, text)

                elapsed = time.time() - t0
                parsed = parse_json(response)
                if not parsed:
                    raise ValueError("Invalid JSON")

                present = parsed.get("hierarchical_interaction_present", "No")
                conf = parsed.get("confidence")
                reasoning = parsed.get("scales_reasoning", {})
                detected_scales = set(reasoning.get("detected_scales", []))
                explanations = reasoning.get("scale_explanations", {})
                causal = parsed.get("causal_pattern", {})
                summary = parsed.get("summary")

                if present == "No":
                    detected_scales = set()
                    causal = {"type": "N/A", "pathway": "N/A"}
                    explanations = {}
                    summary = "N/A"

                model_status[tag] = present
                model_scales[tag] = detected_scales
                rec.update({
                    f"{tag}: Interaction Present": present,
                    f"{tag}: Confidence": conf,
                    f"{tag}: Scales": ", ".join(sorted(detected_scales)) if detected_scales else "N/A",
                    f"{tag}: Causal Type": causal.get("type", "N/A"),
                    f"{tag}: Causal Pathway": causal.get("pathway", "N/A"),
                    f"{tag}: Scale Explanations": "; ".join([f"{k}: {v}" for k, v in explanations.items()]),
                    f"{tag}: Summary": summary,
                })

                print(f"Result: {present} (took {elapsed:.1f}s)")

            except Exception as e:
                model_status[tag] = "Error"
                model_scales[tag] = {"Error"}
                rec.update({f"{tag}: {col}": "Error" for col in [
                    "Interaction Present", "Confidence", "Scales", "Causal Type", 
                    "Causal Pathway", "Scale Explanations", "Summary"
                ]})
                print(f"ERROR: {e}")

        yes_scales = [scales for tag, scales in model_scales.items() if model_status.get(tag) == "Yes"]
        if any(model_status.get(tag) == "Yes" for tag in MODELS):
            valid_statuses = {v for v in model_status.values() if v != "Error"}
            rec["Interaction Present Disagreement"] = "Yes" if len(valid_statuses) > 1 else "No"

            if len(yes_scales) > 1:
                common = set.intersection(*yes_scales)
                total = set.union(*yes_scales)
                disagreement = sorted(total - common)
                rec["Scale Disagreement"] = ", ".join(disagreement) if disagreement else "None"
            else:
                rec["Scale Disagreement"] = "N/A"

            results.append(rec)

    if not results:
        print("\nNo results to write.")
        return

    column_order = ["Source", "Full Paragraph Text", "Interaction Present Disagreement", "Scale Disagreement"]
    for tag in MODELS:
        column_order.extend([
            f"{tag}: Interaction Present", f"{tag}: Confidence", f"{tag}: Scales",
            f"{tag}: Causal Type", f"{tag}: Causal Pathway", f"{tag}: Scale Explanations", f"{tag}: Summary"
        ])

    out_df = pd.DataFrame(results)
    out_df = out_df[[col for col in column_order if col in out_df.columns]]
    out_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(out_df)} rows to {args.output}")

if __name__ == "__main__":
    main()