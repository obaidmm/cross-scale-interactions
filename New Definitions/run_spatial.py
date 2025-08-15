#!/usr/bin/env python3
"""
Detect spatial cross-scale interactions in paragraphs with Claude, OpenAI, and Gemini models.
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
You are an expert analyst identifying **spatial (geographical) cross-scale interactions**. Your task is to analyze the provided text and classify any interaction using the detailed causal typology. Follow all rules precisely and apply the strictest possible interpretation of "interaction".
</task>

<scales>
  <scale name="Local / Site-specific">
    <description>Specific places like neighborhoods, cities, or facilities.</description>
    <examples>A zero-waste policy in a school district; a pollution issue in one town.</examples>
  </scale>
  <scale name="Sub-national / Regional">
    <description>Administrative or functional regions within a country that are larger than a specific local area.</description>
    <examples>Ontario's electricity grid; California climate law.</examples>
  </scale>
  <scale name="National">
    <description>Entire country-level policies, actions, or discourses.</description>
    <examples>Canada's carbon pricing; Korea's industrial development plans.</examples>
  </scale>
  <scale name="Transnational / Cross-border">
    <description>Multi-country regional initiatives, issues that cross borders, or planetary considerations.</description>
    <examples>ASEAN economic zones; EU environmental directives; global tariffs.</examples>
  </scale>
</scales>

<rules>
  <rule id="1" type="CRITICAL">
    **STRICT INTERPRETATION OF CAUSALITY:**
    - **Mechanism is Required:** You must identify a stated mechanism of influence (e.g., a law mandating change, funding enabling action, protests causing a policy response).
    - **Co-location is NOT an Interaction:** Do not flag an interaction where one scale is simply located within another (e.g., "Paris is in France").
    - **Reject Correlation:** Do not infer a link between two events at different scales if the text does not explicitly state one.
  </rule>

  <rule id="2">
    **CLASSIFY INTERACTION TYPE:** If a valid interaction is found, categorize it using the **most specific** definition below.
    - **Direct Causal Link:** An entity at scale A directly causes an effect in an entity at scale B (A -> B).
    - **Causal Chain:** A influences B, which in turn influences C (A -> B -> C).
    - **Convergent Influence:** Multiple entities (e.g., A, B) independently influence a common entity C (A -> C, B -> C).
    - **Bidirectional Link:** Entities at scales A and B mutually influence each other (A <-> B).
    - **Feedback Loop:** A closed loop where effects cycle back to the origin scale (A -> B -> A).
    - **Mixed/Complex Pattern:** A hybrid of multiple types or a complex, non-linear pathway.
  </rule>

  <rule id="3">If `cross_scale_connection_present` is "No", all other detail fields must be "N/A" or empty.</rule>
</rules>

<output_format>
{
  "cross_scale_connection_present": "Yes|No",
  "confidence": 0.0-1.0,
  "scale_reasoning": {
    "detected_scales": ["Local / Site-specific", "National"],
    "scale_explanations": {
      "Local / Site-specific": "Explanation for why an element is local.",
      "National": "Explanation for why an element is national."
    }
  },
  "scale_relationship_type": "Direct Causal Link | Causal Chain | Convergent Influence | Bidirectional Link | Feedback Loop | Mixed/Complex Pattern | N/A",
  "paragraph_summary": "[Local -> National] A summary of how the influence flows between the scales."
}
</output_format>
"""

OPENAI_DETECTOR_PROMPT = r"""
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
   - **Correlation is NOT Causation:** Do not connect events at different scales without a stated link.

3. **Classify the Interaction Type.** If a link meets the strict criteria, classify it using the **most specific** definition below:
    - **Direct Causal Link:** A -> B.
    - **Causal Chain:** A -> B -> C.
    - **Convergent Influence:** A -> C and B -> C.
    - **Bidirectional Link:** A <-> B (explicitly stated mutual influence).
    - **Feedback Loop:** A -> B -> A (a closed causal cycle).
    - **Mixed/Complex Pattern:** A hybrid or other complex structure.

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
**TASK:**
Identify and describe **spatial (geographical) cross-scale interactions** in the provided text. You must only use the four specified spatial scales and classify the interaction based on the defined causal structures.

---

**SPATIAL SCALE DEFINITIONS**

* **Local / Site-specific**: A specific place like a city, neighborhood, or single facility.
* **Sub-national / Regional**: A region within a country, like a province or state.
* **National**: An entire country's policy or action.
* **Transnational / Cross-border**: Actions involving multiple countries or planetary systems.

---

**INSTRUCTIONS & RULES**

1.  **Detect Interaction**: Is there a causal link **explicitly stated** in the text?
    * **Strict Interpretation**: A mechanism of influence (e.g., a law, funding, physical effect) must be stated. Geographical nesting (e.g., "a city in a country") is NOT an interaction.

2.  **Classify Interaction Type**: If a valid interaction is found, categorize the relationship using the **most specific** definition from the list below.
    * **Direct Causal Link**: An entity at scale A directly causes an effect in an entity at scale B (A -> B).
    * **Causal Chain**: An entity at scale A influences B, which *in turn* influences C (A -> B -> C).
    * **Convergent Influence**: Two or more entities from different scales (e.g., A, B) independently influence a common entity at scale C (A -> C, B -> C).
    * **Bidirectional Link**: An entity at scale A and an entity at scale B explicitly and mutually influence each other (A <-> B).
    * **Feedback Loop**: A closed loop where effects eventually cycle back to the originating scale or entity (e.g., A -> B -> A).
    * **Mixed/Complex Pattern**: A hybrid of multiple types above or a non-linear pathway that cannot be simply classified.

3.  **Justify & Summarize**: Briefly justify the scale classifications and write a single-sentence summary explaining the interaction, starting with the directional structure (e.g., `[Local -> National]...`).

4.  **No Interaction**: If no valid interaction is found, set `cross_scale_connection_present` to "No" and all other relevant fields to "N/A".

---

**OUTPUT FORMAT (Return ONLY this JSON object):**

```json
{
  "cross_scale_connection_present": "Yes|No",
  "confidence": 0.0-1.0,
  "scale_reasoning": {
    "detected_scales": ["Local / Site-specific", "National"],
    "scale_explanations": {
      "Local / Site-specific": "Brief justification for the local scale classification.",
      "National": "Brief justification for the national scale classification."
    }
  },
  "scale_relationship_type": "Direct Causal Link | Causal Chain | Convergent Influence | Bidirectional Link | Feedback Loop | Mixed/Complex Pattern | N/A",
  "paragraph_summary": "[Local -> National] A concise summary of the cross-scale interaction."
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
    """
    Main function to run the spatial interaction analysis script.
    It reads paragraphs, analyzes them with multiple LLMs, calculates disagreement,
    and saves the results to a CSV file.
    """
    parser = argparse.ArgumentParser(description="Analyze paragraphs for spatial cross-scale interactions.")
    parser.add_argument("-i", "--input",  default="paragraphs_comparison.json", help="Input JSON file with paragraphs.")
    parser.add_argument("-o", "--output", default="spatial_results.csv", help="Output CSV file for results.")
    parser.add_argument("-n", "--limit",  type=int, default=None, help="Limit the number of paragraphs to process.")
    args = parser.parse_args()

    try:
        paragraphs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.input}")
        return

    if args.limit:
        paragraphs = paragraphs[: args.limit]

    if not paragraphs:
        print("No paragraphs found in the input file. Exiting.")
        return

    df = pd.DataFrame({
        "Full Paragraph Text": [p.get("paragraph", "") for p in paragraphs],
        "Source":              [p.get("pdf", "N/A")   for p in paragraphs],
    })

    results = []
    total = len(df)
    print(f"Starting spatial interaction scan on {total} paragraph(s)…")

    for idx, row in df.iterrows():
        text = row["Full Paragraph Text"]
        if not text:
            continue

        rec = {"Source": row["Source"], "Full Paragraph Text": text}
        model_status = {}
        model_scales = {}

        for tag, mid in MODELS.items():
            # Determine the correct API caller based on the model tag
            if tag.startswith("claude"):
                caller = analyze_claude
            elif tag.startswith("gemini"):
                caller = analyze_gemini
            else:
                caller = analyze_openai

            print(f"[{idx+1}/{total}] Analyzing with {tag}… ", end="", flush=True)
            t0 = time.time()
            try:
                parsed = parse_json(caller(mid, text))
                if not parsed:
                    raise ValueError("could not parse JSON from model response")

                # Extract data from the parsed JSON response
                present = parsed.get("cross_scale_connection_present", "No")
                conf = parsed.get("confidence")
                scale_reasoning = parsed.get("scale_reasoning", {})
                detected_scales_list = scale_reasoning.get("detected_scales", [])
                detected_str = ", ".join(detected_scales_list)
                expl = "; ".join([f"{k}: {v}" for k, v in scale_reasoning.get("scale_explanations", {}).items()])
                relationship_type = parsed.get("scale_relationship_type", "")
                summary = parsed.get("paragraph_summary", "")

                # Store status and detected scales for disagreement analysis
                model_status[tag] = present
                model_scales[tag] = set(detected_scales_list)

                # If no interaction is present, overwrite fields with N/A
                if present == "No":
                  detected_str = expl = relationship_type = summary = "N/A"

                # Update the record for this paragraph
                rec.update({
                      f"{tag}: Interaction Present": present,
                      f"{tag}: Confidence":  conf,
                      f"{tag}: Detected Scales":  detected_str,
                      f"{tag}: Scale Explanations": expl,
                      f"{tag}: Relationship Type": relationship_type,
                      f"{tag}: Summary":     summary,
                  })
                print(f"Result: {present} ({time.time()-t0:.1f}s)")

            except Exception as e:
                print(f"ERROR ({e})")
                # Populate error fields for this model
                for col in ["Interaction Present", "Confidence", "Detected Scales",
                            "Scale Explanations", "Relationship Type", "Summary"]:
                    rec[f"{tag}: {col}"] = "Error"
                model_status[tag] = "Error"
                model_scales[tag] = set()


        # --- Disagreement Calculation ---
        # Keep the record only if at least one model found an interaction
        if any(v == "Yes" for v in model_status.values()):
            # Check for disagreement on whether an interaction is present
            valid_statuses = {v for v in model_status.values() if v != "Error"}
            rec["Interaction Present Disagreement"] = "Yes" if len(valid_statuses) > 1 else "No"

            # Check for disagreement on the specific scales involved
            yes_scales = [scales for tag, scales in model_scales.items() if model_status.get(tag) == "Yes"]
            if len(yes_scales) > 1:
                common = set.intersection(*yes_scales)
                total = set.union(*yes_scales)
                disagreement = sorted(list(total - common))
                rec["Scale Disagreement"] = ", ".join(disagreement) if disagreement else "None"
            else:
                # Not enough data to compare scale disagreement
                rec["Scale Disagreement"] = "N/A"

            results.append(rec)

    # --- Final Output ---
    if not results:
        print("\nNo spatial interactions were detected in any of the paragraphs.")
    else:
        out_df = pd.DataFrame(results)
        print(f"\nWriting {len(out_df)} rows with detected interactions to {args.output}")
        out_df.to_csv(args.output, index=False)



if __name__ == "__main__":
    main()
