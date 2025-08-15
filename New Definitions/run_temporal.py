#!/usr/bin/env python3
"""
Detect temporal cross-scale interactions in paragraphs with Claude, OpenAI, and Gemini models.
then write the structured results to a CSV.
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import anthropic
from openai import OpenAI
from google import genai


import config   

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
You are an expert analyst identifying **temporal cross-scale interactions**. Your task is to analyze the provided text, determine if a strict causal link exists between any two of the three defined time scales, and classify the interaction's structure.
</task>

<scales>
  <scale name="Short-Term">
    <description>Effects unfolding over days to 1 year.</description>
    <examples>Quarterly profits, a new product launch, a temporary hiring freeze.</examples>
  </scale>
  <scale name="Medium-Term">
    <description>Covers multi-year transitions or strategies, from greater than 1 year up to 5 years.</description>
    <examples>A 3-year climate adaptation plan; Phasing in new sustainability reporting standards over four years.</examples>
  </scale>
  <scale name="Long-Term">
    <description>Processes unfolding over decades (greater than 5 years). Includes any language that implies 'long term' if no explicit timeframe is mentioned.</description>
    <examples>Rising global temperatures; Population aging; The depreciation and replacement cycle of public infrastructure.</examples>
  </scale>
</scales>

<rules>
  <rule id="1" type="CRITICAL">
    **STRICT CAUSALITY RULES:**
    - **Mechanism is Required:** The text must explicitly state *how* one timeframe influences another (e.g., "the annual budget cuts *led to* a decade-long decline in service quality").
    - **Reject Co-occurrence:** Do not flag an interaction if different timeframes are merely mentioned together without a stated causal link.
  </rule>

  <rule id="2">
    **CLASSIFY INTERACTION TYPE:** If a valid interaction is found, categorize it using the **most specific** definition below.
    - **Direct Causal Link:** An event at one time scale directly causes an effect at another (e.g., Short-term -> Long-term).
    - **Causal Chain:** A sequence of influence across three or more events (e.g., Short-term -> Medium-term -> Long-term).
    - **Convergent Influence:** Multiple events influence a single event at another time scale.
    - **Bidirectional Link:** Two time scales mutually influence each other (e.g., Short-term <-> Long-term).
    - **Feedback Loop:** A closed loop where effects cycle back to the origin (e.g., Short-term -> Long-term -> Short-term).
    - **Mixed/Complex Pattern:** A hybrid of multiple types or a complex, non-linear pathway.
  </rule>

  <rule id="3">If "temporal_interaction_present" is "No", all other detail fields must be "N/A" or empty.</rule>
</rules>

<output_format>
{
  "temporal_interaction_present": "Yes|No",
  "confidence": 0.0-1.0,
  "timeframe_reasoning": {
    "detected_timeframes": ["Short-term", "Long-term"],
    "timeframe_explanations": {
      "Short-term": "Explanation for the short-term element.",
      "Long-term": "Explanation for the long-term element."
    }
  },
  "scale_relationship_type": "Direct Causal Link | Causal Chain | Convergent Influence | Bidirectional Link | Feedback Loop | Mixed/Complex Pattern | N/A",
  "summary": "[Short-term -> Long-term] A summary of the causal link, with the prefix reflecting the primary direction of influence."
}
</output_format>
"""

OPENAI_DETECTOR_PROMPT=r"""
/*
Role: Geospatial Systems Analyst

Your task is to detect and describe **spatial cross-scale interactions** using the 4-level geographical scheme and detailed causal typology provided. Adhere strictly to all definitions and criteria.

────────────────────────────────────
I. SPATIAL SCALE DEFINITIONS (Use ONLY these)
• Local / Site-specific:      A specific city, facility, or neighborhood.
• Sub-national / Regional:    A state, province, or large internal region.
• National:                   An entire country.
• Transnational / Cross-border: Multiple countries or planetary systems.

────────────────────────────────────
II. ANALYSIS & CLASSIFICATION

1. **Find an Explicitly Stated Causal Link.** The text must state *how* one scale influences another.

2. **CRITICAL: Strict Interaction Criteria:**
   - **Mechanism Required:** The text must describe a mechanism (e.g., policy, budget, protest, physical flow). No described mechanism means no interaction.
   - **Co-location is NOT an Interaction:** A city being inside a country is a geographical fact, not an interaction for this task.

3. **INTERACTION TYPE DEFINITIONS (Classify using the most specific type):**
    - **Direct Causal Link:** An entity at scale A directly causes an effect in an entity at scale B (A -> B).
    - **Causal Chain:** An entity at scale A influences B, which in turn influences C (A -> B -> C).
    - **Convergent Influence:** Multiple entities (e.g., A, B) independently influence a common entity C (A -> C, B -> C).
    - **Bidirectional Link:** Entities at scales A and B mutually influence each other (A <-> B).
    - **Feedback Loop:** A closed loop where effects cycle back to the origin scale (A -> B -> A).
    - **Mixed/Complex Pattern:** A hybrid of multiple types or a complex, non-linear pathway.

4. Provide brief justifications for your scale classifications and a one-sentence summary.

5. If no interaction exists, set `cross_scale_connection_present` to "No" and other fields to "N/A".

────────────────────────────────────
III. OUTPUT (Return a single JSON object, no code fences):
{
  "cross_scale_connection_present": "Yes|No",
  "confidence": 0.0-1.0,
  "scale_reasoning": {
    "detected_scales": ["Local / Site-specific", "National"],
    "scale_explanations": {
      "Local / Site-specific": "Justification for local scale.",
      "National": "Justification for national scale."
    }
  },
  "scale_relationship_type": "Direct Causal Link | Causal Chain | Convergent Influence | Bidirectional Link | Feedback Loop | Mixed/Complex Pattern | N/A",
  "paragraph_summary": "[Local -> National] Summary of the interaction."
}
*/
"""

GEMINI_DETECTOR_PROMPT = r"""
<task>
You are an expert analyst tasked with identifying **temporal cross-scale interactions**. Your goal is to find explicit causal links between any two of the three defined time scales and classify the structure of that link using the detailed typology provided.
</task>

---
**TEMPORAL SCALE DEFINITIONS**

* **Short-Term**: Effects unfolding over days to 1 year.
    * *Examples*: Quarterly profits, a new product launch, a temporary hiring freeze.
* **Medium-Term**: Covers multi-year transitions or strategies (greater than 1 year up to 5 years).
    * *Examples*: A 3-year climate adaptation plan; Phasing in new sustainability reporting standards.
* **Long-Term**: Processes unfolding over decades (greater than 5 years). Includes any words that imply ‘long term’ if the time frame is not mentioned explicitly.
    * *Examples*: Rising global temperatures; Population aging; The long-term depreciation and replacement cycle of public infrastructure.

---
**MANDATORY: Strict Interpretation of "Interaction"**

1.  **A Mechanism of Influence Must Be Stated:** The text must describe *how* an event in one time scale affects another (e.g., a short-term action that *contributes to* a long-term outcome). If no mechanism is stated, there is no interaction.
2.  **Reject Association as Causation:** Do not flag an interaction if events in different time scales are simply mentioned together without a direct causal statement.

---
**INSTRUCTIONS**

1.  **Analyze**: Determine if a temporal interaction that meets the strict criteria exists between any two of the three time scales.
2.  **Classify Interaction Type**: If yes, categorize the relationship using the **most specific** definition from the list below.
    * **Direct Causal Link:** An event at one time scale directly causes an effect at another.
    * **Causal Chain:** A sequence of influence across three or more events (e.g., Short-term -> Medium-term -> Long-term).
    * **Convergent Influence:** Multiple events influencing a single event at another time scale.
    * **Bidirectional Link:** A stated two-way, mutual influence (e.g., Short-term <-> Long-term).
    * **Feedback Loop:** A closed causal cycle (e.g., Short-term -> Long-term -> Short-term).
    * **Mixed/Complex Pattern:** A hybrid or complex non-linear structure.
3.  **Summarize**: Write a one-sentence summary beginning with a bracketed prefix (e.g., `[Short-term -> Long-term]...`) that captures the main flow of the interaction.
4.  **No Interaction**: If none is found, `temporal_interaction_present` must be "No", and all other fields "N/A".

---
**OUTPUT FORMAT (Return ONLY this JSON object):**
```json
{
  "temporal_interaction_present": "Yes|No",
  "confidence": 0.0-1.0,
  "timeframe_reasoning": {
    "detected_timeframes": ["Short-term", "Medium-term"],
    "timeframe_explanations": {
      "Short-term": "Brief justification for the short-term classification.",
      "Medium-term": "Brief justification for the medium-term classification."
    }
  },
  "scale_relationship_type": "Direct Causal Link | Causal Chain | Convergent Influence | Bidirectional Link | Feedback Loop | Mixed/Complex Pattern | N/A",
  "summary": "[Short-term -> Medium-term] A concise summary of the cross-scale temporal interaction."
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
    parser.add_argument("-o", "--output", default="temporal_results.csv")
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
    print(f"Starting temporal interaction scan on {total} paragraph(s)…")

    for idx, row in df.iterrows():
        text = row["Full Paragraph Text"]
        rec  = {"Source": row["Source"], "Full Paragraph Text": text}
        model_status = {}

        for tag, mid in MODELS.items():
            if tag.startswith("claude"):
                caller = analyze_claude
            elif tag.startswith("gemini"):
                    caller = analyze_gemini
            else:
                    caller = analyze_openai
            print(f"[{idx+1}/{total}] {tag} … ", end="", flush=True)
            t0 = time.time()
            try:
                parsed = parse_json(caller(mid, text))
                if not parsed:
                    raise ValueError("could not parse JSON")

                present = parsed.get("temporal_interaction_present", "No")
                conf    = parsed.get("confidence")
                tf      = parsed.get("timeframe_reasoning", {})
                detected = ", ".join(tf.get("detected_timeframes", []))
                expl    = "; ".join([f"{k}: {v}" for k, v in tf.get("timeframe_explanations", {}).items()])
                direction  = parsed.get("direction", "")
                int_type   = parsed.get("interaction_type", "")
                summary    = parsed.get("summary", "")

                if present == "No":
                    detected = expl = direction = int_type = summary = "N/A"

                rec.update({
                    f"{tag}: Interaction": present,
                    f"{tag}: Confidence":  conf,
                    f"{tag}: Timeframes":  detected,
                    f"{tag}: Explanations": expl,
                    f"{tag}: Direction":   direction,
                    f"{tag}: Type":        int_type,
                    f"{tag}: Summary":     summary,
                })
                model_status[tag] = present
                print(f"{present} ({time.time()-t0:.1f}s)")
            except Exception as e:
                print("ERROR")
                for col in ["Interaction", "Confidence", "Timeframes",
                            "Explanations", "Direction", "Type", "Summary"]:
                    rec[f"{tag}: {col}"] = "Error"
                model_status[tag] = "Error"

        # keep paragraph only if at least one model says "Yes"
        if any(v == "Yes" for v in model_status.values()):
            rec["Temporal Interaction Found"] = "Yes"
            rec["Disagreement"] = "Yes" if len(set(model_status.values())) > 1 else "No"
            results.append(rec)

    out_df = pd.DataFrame(results)
    if out_df.empty:
        print("No temporal interactions detected.")
    else:
        print(f"Writing {len(out_df)} rows to {args.output}")
        out_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
