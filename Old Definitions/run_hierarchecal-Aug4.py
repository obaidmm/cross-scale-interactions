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
claude = anthropic.Client(api_key=config.ANTHROPIC_API_KEY)
client       = OpenAI(api_key=config.OPENAI_API_KEY)
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
You are an expert analyst identifying and classifying **hierarchical cross-scale interactions** in text.
A hierarchical interaction exists when an entity at one scale *clearly* affects, or is affected by, an entity at another, distinct scale.
</task>

<scales>
  <scale name="Individual">
    <description>Involves specific individuals or informal groups like workers, employees, customers, or household members.</description>
    <examples>A worker's job dissatisfaction; An individual consumer’s boycott.</examples>
  </scale>
  <scale name="Organizational">
    <description>Involves a single organization (e.g., a company, NGO, government agency, or a brand) or its internal units. This applies even if the organization operates across borders. Also includes formal groups like worker associations.</description>
    <examples>A company’s ESG strategy; A hospital conducting a waste audit; A specific energy supplier.</examples>
    <clarification>'Community members' are generally not at this scale; see 'Environmental and Societal'. The UN as a single entity is considered Organizational.</clarification>
  </scale>
  <scale name="Inter-Organizational / Sector">
    <description>Involves groups of organizations and their interactions, like public-private partnerships, industry alliances, or entire supply chains treated as a system.</description>
    <examples>Cross-sector supply chains; Industry lobbying coalitions; The UN ecosystem of partnerships.</examples>
    <clarification>Distinguish this from a single organization. For example, 'the supply chain' (Inter-Organizational) vs. a specific 'supplier' (Organizational).</clarification>
  </scale>
  <scale name="Environmental and Societal">
    <description>Involves societal-level systems or institutional structures. Includes capital markets, governments, communities, religious organizations, and natural systems.</description>
    <examples>Government regulation; Capital market norms; A forest ecosystem limiting access to land.</examples>
    <clarification>'Community' as a whole or abstract entity belongs here.</clarification>
  </scale>
</scales>

<causal_patterns>
  <pattern name="Direct Causal Link">A -> B. An entity at scale A causes a direct effect in an entity at scale B.</pattern>
  <pattern name="Causal Chain">A -> B -> C. An entity at scale A influences B, which then influences C.</pattern>
  <pattern name="Convergent Influence">[A, B] -> C. Multiple entities at different scales (A, B) independently influence a common entity (C).</pattern>
  <pattern name="Bidirectional Link">A <-> B. Two entities at different scales mutually influence each other.</pattern>
  <pattern name="Feedback Loop">A -> B -> A. A closed loop where effects eventually cycle back to the originating scale or entity.</pattern>
  <pattern name="Mixed/Complex Pattern">A hybrid or multi-directional pattern that combines other types.</pattern>
</causal_patterns>

<rules>
  - Code based on an activity by an actor, not just a topic.
  - An interaction requires a clear link between at least two different scales.
</rules>

<output_instructions>
  1. "hierarchical_interaction_present": "Yes" or "No".
  2. "confidence": A float from 0.0 (low) to 1.0 (high).
  3. "scales_reasoning":
     - "detected_scales": List the scales found (e.g., ["Individual", "Organizational"]).
     - "scale_explanations": Briefly justify each scale's classification.
  4. "causal_pattern":
     - "type": The name of the causal pattern from the list (e.g., "Causal Chain").
     - "pathway": The string representing the causal flow (e.g., "Individual → Organizational → Environmental and Societal").
  5. "summary": A compact sentence starting with the bracketed pathway, e.g., "[Individual → Organizational] An employee's whistleblowing forces a change in corporate policy."
  
  - If "hierarchical_interaction_present" == "No", the other fields can be "N/A" or empty.
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
You are an analyst detecting and classifying **hierarchical cross-scale interactions**.
A hierarchical interaction occurs when an entity at one scale clearly influences, or is influenced by, an entity at another scale.

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
RULES
- Focus on the actors and their activities, not just the topic.
- An interaction requires a link between at least two different scales.
- If no interaction, flag "No" and leave other fields as "N/A".

────────────────────────────────────
WHAT TO RETURN (pure JSON, no code fences):
1. hierarchical_interaction_present: "Yes" or "No"
2. confidence: 0.0 - 1.0
3. scales_reasoning:
   - detected_scales: List of scales found (e.g., ["Individual", "Organizational"]).
   - scale_explanations: A mapping of scale -> brief justification.
4. causal_pattern:
   - type: The name of the causal pattern identified (e.g., "Direct Causal Link").
   - pathway: The string showing the causal flow (e.g., "Individual → Organizational").
5. summary:
   - One concise sentence starting with the bracketed pathway from the causal_pattern.
   - Example: "[Individual → Organizational] An employee's whistleblowing prompts a new corporate policy."

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
You are an analyst detecting and classifying **hierarchical cross-scale interactions**.
A hierarchical interaction happens when an entity at one scale clearly influences, or is influenced by, an entity at another scale.

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
**RULES**

1.  **Focus on Actors**: Code based on an activity tied to an actor, not just a general topic.
2.  **Interaction Required**: An interaction requires a clear link between at least two different scales.
3.  **Output**: Respond with a single, valid JSON object only.

---
**OUTPUT INSTRUCTIONS**

1.  `hierarchical_interaction_present`: "Yes" or "No".
2.  `confidence`: Float from 0.0 to 1.0.
3.  `scales_reasoning`:
    * `detected_scales`: A list of the scales you found.
    * `scale_explanations`: A dictionary explaining why the text fits each identified scale.
4.  `causal_pattern`: An object containing:
    * `type`: The name of the pattern from the definitions (e.g., "Feedback Loop").
    * `pathway`: The string representing the causal flow (e.g., "Organizational -> Environmental and Societal -> Organizational").
5.  `summary`: A single sentence summarizing the interaction, starting with the bracketed pathway. Example: `"[Individual -> Organizational] An employee's whistleblowing forces a change in corporate policy."`

If `hierarchical_interaction_present` is "No", set other fields to "N/A" or empty.

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
    "pathway": "Scale A -> Scale B -> Scale C"
  },
  "summary": "[Scale A -> Scale B -> Scale C] ..."
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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",  default="paragraphs_comparison.json")
    parser.add_argument("-o", "--output", default="hierarchical_results.csv")
    parser.add_argument("-n", "--limit",  type=int, default=None)
    args = parser.parse_args()

    paragraphs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.limit:
        paragraphs = paragraphs[: args.limit]

    if not paragraphs:
        print("No paragraphs found, exiting.")
        return

    df = pd.DataFrame({
        "Full Paragraph Text": [p["paragraph"] for p in paragraphs],
        "Source":              [p["pdf"]       for p in paragraphs],
    })

    results = []
    total = len(df)
    print(f"Starting hierarchical interaction scan on {total} paragraph(s)…")

    for idx, row in df.iterrows():
        text = row["Full Paragraph Text"]
        rec  = {"Source": row["Source"], "Full Paragraph Text": text}
        model_status = {}

        for tag, mid in MODELS.items():
            if tag.startswith("claude"):
                caller = analyze_claude
            elif tag.startswith("gemini"):
                caller = analyze_gemini
            else: # openai
                caller = analyze_openai
            
            print(f"[{idx+1}/{total}] {tag} … ", end="", flush=True)
            t0 = time.time()
            try:
                raw_response = caller(mid, text)
                parsed = parse_json(raw_response)
                
                if not parsed:
                    raise ValueError("could not parse JSON")

                present = parsed.get("hierarchical_interaction_present", "No")
                conf    = parsed.get("confidence")
                reasoning = parsed.get("scales_reasoning", {})
                detected = ", ".join(reasoning.get("detected_scales", []))
                expl = "; ".join([f"{k}: {v}" for k, v in reasoning.get("scale_explanations", {}).items()])
                direction = parsed.get("direction", "")
                summary = parsed.get("summary", "")

                if present == "No":
                    detected = expl = direction = int_type = summary = "N/A"

                rec.update({
                f"{tag}: Interaction Present": present,
                f"{tag}: Confidence":      conf,
                f"{tag}: Scales":          detected,
                f"{tag}: Scale Explanations": expl,
                f"{tag}: Direction":       direction,
                f"{tag}: Summary":         summary,
                })
                model_status[tag] = present
                print(f"{present} ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"ERROR ({e})")
                for col in ["Interaction Present", "Confidence", "Scales",
                            "Scale Explanations", "Direction", "Summary"]:
                    rec[f"{tag}: {col}"] = "Error"
                model_status[tag] = "Error"

        # keep paragraph only if at least one model says "Yes"
        if any(v == "Yes" for v in model_status.values()):
          valid_statuses = {v for v in model_status.values() if v != "Error"}
          rec["Disagreement"] = "Yes" if len(valid_statuses) > 1 else "No"
          results.append(rec)


    out_df = pd.DataFrame(results)
    if out_df.empty:
        print("No hierarchical interactions detected.")
    else:
        print(f"Writing {len(out_df)} rows to {args.output}")
        out_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
