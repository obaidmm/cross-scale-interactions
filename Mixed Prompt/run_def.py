#!/usr/bin/env python3
"""
Detect hierarchical (structural) cross-scale interactions in text using
Claude, OpenAI, and Gemini models, then write the structured results to a CSV.
The prompts are built to strictly adhere to the user-provided rule set.
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
    "gemini2.0":    "gemini-2.0-flash"
}


CLAUDE_DETECTOR_PROMPT = r"""
<task>
You are an expert systems analyst. Your job is to find **cross-scale interactions** in the user-supplied paragraph.  
Follow the definitions and rules below exactly.
</task>

<interaction_types>
  <type name="Direct Causal Link"> An entity at scale A causes an effect in an entity at scale B (A -> B)</type>
  <type name="Causal Chain">A -> B -> C. A sequential link with three or more entities that spans at least two distinct scales.</type>
  <type name="Convergent Influence">Entities at different scales (A, B) each influence a common entity at scale C (A -> C, B -> C).</type>
  <type name="Bidirectional Link">Two entities at different scales influence each other (A <-> B).</type>
  <type name="Feedback Loop">A closed loop where effects circle back to the origin (A -> B -> A or A -> B -> C -> A) and the entities sit at different scales.</type>
  <type name="Mixed/Complex Pattern">A hybrid or non-linear structure that combines more than one of the patterns above.</type>
</interaction_types>

<scale_definitions>
  <dimension name="I. HIERARCHICAL (STRUCTURAL) SCALE">
    <scale name="Individual">Specific people or informal groups such as workers, customers, or family members.</scale>
    <scale name="Organizational">A single organization or a formal unit inside it (company, NGO, agency, brand, union, etc.).</scale>
    <scale name="Inter-Organizational / Sector">Groups of organizations, industry alliances, supply chains with multiple firms, or monolithic social groups such as “citizens” or “communities.”</scale>
    <scale name="Environmental and Societal">Societal-level systems or natural systems (capital markets, government regulation, forest ecosystem).</scale>
  </dimension>

  <dimension name="II. SPATIAL (GEOGRAPHICAL) SCALE">
    <scale name="Local / Site-specific">A neighbourhood, city, or facility.</scale>
    <scale name="Sub-national / Regional">A province, state, or comparable region within a country.</scale>
    <scale name="National">An entire country.</scale>
    <scale name="Transnational / Cross-border">Multi-country or planetary contexts.</scale>
  </dimension>

  <dimension name="III. TEMPORAL SCALE">
    <scale name="Short-Term">Days up to 1 year.</scale>
    <scale name="Medium-Term">Longer than 1 year up to 5 years.</scale>
    <scale name="Long-Term">More than 5 years or any language that implies “long term.”</scale>
  </dimension>
</scale_definitions>

<rules>
- Code only actor-based activities (not abstract topics).
- Identify an interaction only when one entity influences another.
- For each entity, determine its Hierarchical, Spatial, and Temporal scale, but STRICTLY REMEMBER that not all entities will have all three scales.
- Compare these scales across all involved entities.
- Report both:
- (a) For `combined_chain`, provide a concise summary of the single most important interaction path, whether it's within or across dimensions (e.g., "Short-Term -> Long-Term" or "National -> Transnational / Cross-border"). An interaction path should not involve more than three distinct entities.
  (b) For `scale_change`, detail all dimension-level transitions that show a change.
(c) For `final_scale`, return ONE interaction path showing scale change within a **single dimension only** (either Hierarchical, or Spatial, or Temporal — never a mix).
X Do NOT mix dimensions like "Organizational -> Environmental and Societal -> Long-Term" — this is invalid.
✅ Examples of valid final_scale paths:
- "Short-Term -> Long-Term" (Temporal)
- "Organizational -> Inter-Organizational / Sector -> Environmental and Societal" (Hierarchical)
- "Local / Site-specific -> National -> Transnational / Cross-border" (Spatial)
Always return the most important path **within one dimension**.
- Use "N/A" for any scale not explicitly present in the text.
- If no interaction is found, return: `"interaction_present": "No"` and `"entities": []`.
- Only count an interaction as cross-scale if both the source and target entities have valid, non-N/A scales **within the same dimension** (e.g., Hierarchical or Temporal).
- Do NOT report a scale chain like "Organizational -> N/A". That is not a valid scale transition.
- If one of the entities is missing all valid scales, no cross-scale interaction is present.
- If `"interaction_present": "Yes"`, the `"reasoning"` field is REQUIRED.
  • It must explain why the selected `final_scale` is valid (based on scale change within a single dimension).
  • It must justify the chosen `loop_nature` using clear evidence from the paragraph.
- If `"interaction_present": "No"`, omit `"reasoning"`.
</rules>

<output_format>
{
  "interaction_present": "Yes|No",
  "confidence": 0.0-1.0,
  "entities": [
    {
      "entity_description": "...",
      "hierarchical_scale": "...",
      "spatial_scale": "...",
      "temporal_scale": "..."
    }
  ],
  "relationship": {
    "loop_nature": "Direct Causal Link | Causal Chain | Convergent Influence | Bidirectional Link | Mixed/Complex Pattern | Feedback Loop | N/A",
    "summary": "..."
  },
  "scale_chains": [
    {
      "from_entity": INT,
      "to_entity": INT,
      "scales_crossed": ["Hierarchical", "Temporal", "Spatial"],
      "scale_change": {
        "Hierarchical": "Organizational -> Environmental and Societal",
        "Temporal": "Short-Term -> Long-Term",
        "Spatial": "Local / Site-specific -> National"
      },
      "combined_chain": "Organizational -> - Individual"
    }
  ],
  "final_scale": "Short-Term -> Long-Term",
  "reasoning:": "This must be included if interaction_present is 'Yes'. Clearly justify the scale change and causal interaction."
}
</output_format>
"""


OPENAI_DETECTOR_PROMPT = r"""
ROLE: Multi-Dimensional Systems Analyst

TASK: Identify cross-scale causal interactions in the paragraph using strict definitions. 
Important: A valid cross-scale interaction must occur **within a single scale dimension only** (Hierarchical OR Spatial OR Temporal — not a mix). 
Return a single JSON object.

INTERACTION TYPES:
- Direct Causal Link: An entity at scale A causes an effect in an entity at scale B (A -> B)
- Causal Chain: A -> B -> C. A sequential relationship where an entity at scale A influences an entity at scale B, 
which then influences an entity at scale C (involves three or more entities across at least two scales).
- Convergent Influence: Multiple entities at different scales (A, B) independently influence an entity (or entities) 
at a common scale C (A -> C, B -> C)
- Bidirectional Link: An entity at scale A and an entity at scale B mutually influence each other (A <-> B), 
forming a two-way causal relationship across scales.
- Feedback Loop: A closed loop of causality involving two or more different entities at different scales, 
where effects eventually cycle back to the originating entity (e.g., A -> B -> A or A -> B -> C -> A).
- Mixed/Complex Pattern: A hybrid configuration involving multiple types of causal relationships or non-linear, 
multi-directional pathways across scales.


SCALE DEFINITIONS:
HIERARCHICAL:
- Individual: People, workers, customers
- Organizational: One org or department
- Inter-Organizational / Sector: Coalitions, supply chains, citizen groups
- Environmental and Societal: Natural systems, governments, institutions

SPATIAL:
- Local / Site-specific: Cities, buildings
- Sub-national / Regional: Province or state
- National: Whole country
- Transnational / Cross-border: Multi-country or planetary

TEMPORAL:
- Short-Term: ≤1 year
- Medium-Term: 1-5 years
- Long-Term: >5 years or implied duration

RULES:
- Only code actor-based interactions (not abstract topics).
- Do NOT identify an interaction unless:
  • One entity clearly influences another
  • Both entities have valid (non-N/A) scale values in the same dimension
  • There is **actual scale change** in that dimension
- Redundant transitions like "Organizational -> Organizational" are NOT valid scale changes.
- Do NOT output `final_scale` values that are just a single label (e.g., "Long-Term" or "Organizational")
- Do NOT include "N/A" or any mix of scale dimensions (e.g., "Organizational -> Long-Term") in `final_scale`
- Only accept valid paths with at least 2 levels of scale change in the **same dimension**:
    ✅ "Short-Term -> Long-Term" (Temporal)
    ✅ "Organizational -> Inter-Organizational -> Environmental and Societal" (Hierarchical)
    ✅ "Local -> National -> Transnational" (Spatial)
    ❌ "Organizational -> Organizational"
    ❌ "Long-Term"
    ❌ "Organizational -> Long-Term"
- If `"interaction_present": "Yes"`, the `"reasoning"` field is REQUIRED.
  • It must explain why the selected `final_scale` is valid (based on scale change within a single dimension).
  • It must justify the chosen `loop_nature` using clear evidence from the paragraph.
- If `"interaction_present": "No"`, omit `"reasoning"`.
- If no valid dimension-level transition occurs, return:

OUTPUT FORMAT:
```json
{
  "interaction_present": "Yes|No",
  "confidence": 0.0-1.0,
  "entities": [
    {
      "entity_description": "...",
      "hierarchical_scale": "...",
      "spatial_scale": "...",
      "temporal_scale": "..."
    }
  ],
  "relationship": {
    "loop_nature": "Direct Causal Link | Causal Chain | Convergent Influence | Bidirectional Link | Mixed/Complex Pattern | Feedback Loop | N/A",
    "summary": "..."
  },
  "scale_chains": [
    {
      "from_entity": INT,
      "to_entity": INT,
      "scales_crossed": ["Hierarchical", "Temporal", "Spatial"],
      "scale_change": {
        "Hierarchical": "Organizational -> Environmental and Societal",
        "Temporal": "Short-Term -> Long-Term",
        "Spatial": "Local / Site-specific -> National"
      },
      "combined_chain": "Organizational -> - Individual"
    }
  ],
  "final_scale": "Short-Term -> Long-Term",
  "reasoning:": "This must be included if interaction_present is 'Yes'. Clearly justify the scale change and causal interaction."
}
```
"""


GEMINI_DETECTOR_PROMPT = r"""
**TASK:** Identify cross-scale causal interactions using strict definitions.

X An interaction is **NOT valid** unless:
- BOTH source and target entities have **valid, non-N/A scales**
- The transition happens in **only one scale dimension** (Hierarchical OR Spatial OR Temporal)
- You can describe a scale transition in `scale_change` and `final_scale` **within that one dimension**

**INTERACTION TYPES:**
- Direct Causal Link: A -> B
- Causal Chain: A -> B -> C (≥3 entities, 2+ scales)
- Convergent Influence: A -> C, B -> C
- Bidirectional Link: A <-> B
- Feedback Loop: A -> B -> A (or similar)
- Mixed/Complex Pattern: Hybrid/non-linear paths

**SCALE DEFINITIONS:**
*Hierarchical:*
- Individual: Named people, households
- Organizational: One org or department
- Inter-Organizational: Alliances, families, citizens
- Environmental and Societal: Ecosystems, govts, norms

*Spatial:*
- Local: Town, building, city
- Regional: Province, state
- National: Country-level
- Transnational: Multi-country, global

*Temporal:*
- Short-Term: ≤1 year
- Medium-Term: 1-5 years
- Long-Term: >5 years or implied

**RULES**
- Code only actor-based activities (not abstract topics).
- Identify an interaction only when one entity influences another.
- For each entity, determine its Hierarchical, Spatial, and Temporal scale, but STRICTLY REMEMBER that not all entities will have all three scales.
- Compare these scales across all involved entities.
- Report both:
- (a) For `combined_chain`, provide a concise summary of the single most important interaction path, whether it's within or across dimensions (e.g., "Short-Term -> Long-Term" or "National -> Transnational / Cross-border"). An interaction path should not involve more than three distinct entities.
  (b) For `scale_change`, detail all dimension-level transitions that show a change.
(c) For `final_scale`, return ONE interaction path showing scale change within a **single dimension only** (either Hierarchical, or Spatial, or Temporal — never a mix).
❌ Do NOT mix dimensions like "Organizational -> Environmental and Societal -> Long-Term" — this is invalid.
Examples of valid final_scale paths:
- "Short-Term -> Long-Term" (Temporal)
- "Organizational -> Inter-Organizational / Sector -> Environmental and Societal" (Hierarchical)
- "Local / Site-specific -> National -> Transnational / Cross-border" (Spatial)
Always return the most important path **within one dimension**.
- Use "N/A" for any scale not explicitly present in the text.
- If no interaction is found, return: `"interaction_present": "No"` and `"entities": []`.
- Only count an interaction as cross-scale if both the source and target entities have valid, non-N/A scales **within the same dimension** (e.g., Hierarchical or Temporal).
- Do NOT report a scale chain like "Organizational -> N/A". That is not a valid scale transition.
- If one of the entities is missing all valid scales, no cross-scale interaction is present.

**Examples of INVALID `final_scale`:**
- "Organizational -> Long-Term" X (mixed dimensions)
- "Organizational -> N/A" X (missing scale)

- If `"interaction_present": "Yes"`, the `"reasoning"` field is REQUIRED.
  • It must explain why the selected `final_scale` is valid (based on scale change within a single dimension).
  • It must justify the chosen `loop_nature` using clear evidence from the paragraph.
- If `"interaction_present": "No"`, omit `"reasoning"`.


**OUTPUT FORMAT:**
```json
{
  "interaction_present": "Yes|No",
  "confidence": 0.0-1.0,
  "entities": [
    {
      "entity_description": "...",
      "hierarchical_scale": "...",
      "spatial_scale": "...",
      "temporal_scale": "..."
    }
  ],
  "relationship": {
    "loop_nature": "Direct Causal Link | Causal Chain | Convergent Influence | Bidirectional Link | Mixed/Complex Pattern | Feedback Loop | N/A",
    "summary": "..."
  },
  "scale_chains": [
    {
      "from_entity": INT,
      "to_entity": INT,
      "scales_crossed": ["Hierarchical", "Temporal", "Spatial"],
      "scale_change": {
        "Hierarchical": "Organizational -> Environmental and Societal",
        "Temporal": "Short-Term -> Long-Term",
        "Spatial": "Local / Site-specific -> National"
      },
      "combined_chain": "Organizational -> - Individual"
    }
  ],
  "final_scale": "Short-Term -> Long-Term",
  "reasoning:": "This must be included if interaction_present is 'Yes'. Clearly justify the scale change and causal interaction."
}
```
"""

def analyze_claude(model_id: str, paragraph: str) -> str:
    """Call Claude with the temporal prompt."""
    resp = claude.messages.create(
        model=model_id,
        system=CLAUDE_DETECTOR_PROMPT,
        messages=[{"role": "user", "content": paragraph.strip()}],
        max_tokens=2048,
        temperature=0
    )
    return resp.content[0].text


def analyze_openai(model, text):
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        # max_tokens=600,
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


def parse_json(raw: str):
    if not isinstance(raw, str):
        return None
    # strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*|```$", "", raw.strip(), flags=re.I)
    raw = raw.replace("“", '"').replace("”", '"')
    obj_match = re.search(r"{.*}", raw, flags=re.S)
    if not obj_match:
        return None
    json_str = re.sub(r",\s*([}\]])", r"\\1", obj_match.group(0))
    return json.loads(json_str)

        
# Also need to figure out how to format for bidirectional links 

def format_scale_chains(scale_chains, loop_nature):
    if not scale_chains:
        return "N/A"

    arrow = "<->" if loop_nature == "Bidirectional Link" else "->"

    out = []
    for ch in scale_chains:
        path = f"{ch.get('from_entity','?')} {arrow} {ch.get('to_entity','?')}: "
        out.append(path + ch.get('combined_chain',''))
    return " || ".join(out)


def format_scale_chains_pretty(scale_chains: list) -> str:
    """Formats the scale_chains list into a human-readable string for CSVs."""
    if not scale_chains:
        return "N/A"

    pretty_parts = []
    for chain in scale_chains:
        try:
            from_idx = chain.get("from_entity", "?")
            to_idx = chain.get("to_entity", "?")
            path_str = f"Path({from_idx}->{to_idx})"

            changes = []
            # Extract only the dimensions that actually changed
            for dim, change in chain.get("scale_change", {}).items():
                if "->" in str(change):
                    changes.append(f"{dim}: {change}")

            if changes:
                pretty_parts.append(f"{path_str} | {'; '.join(changes)}")
        except Exception:
            continue
            
    return " || ".join(pretty_parts) if pretty_parts else "N/A"


def format_entities_pretty(entities: list) -> str:
    """Formats a list of entity dicts into a single readable string."""
    if not entities:
        return "N/A"

    summary_parts = []
    for i, entity in enumerate(entities):
        desc = entity.get("entity_description", "N/A")
        h = entity.get("hierarchical_scale", "?")
        s = entity.get("spatial_scale", "?")
        t = entity.get("temporal_scale", "?")
        # Create a compact summary for each entity
        summary_parts.append(f"[{i}] {desc} (H:{h}|S:{s}|T:{t})")

    return " || ".join(summary_parts)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="paragraphs.json")
    parser.add_argument("-o", "--output", default="hierarchical_results.csv")
    parser.add_argument("-n", "--limit", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found. Exiting.")
        return

    paragraphs = json.loads(input_path.read_text(encoding="utf-8"))
    if args.limit:
        paragraphs = paragraphs[:args.limit]

    if not paragraphs:
        print("No paragraphs found, exiting.")
        return

    df = pd.DataFrame(paragraphs)
    results = []
    total = len(df)
    max_retries = 3
    print(f"Scanning {total} paragraph(s) for interactions...")

    for idx, row in df.iterrows():
        text = row["paragraph"]
        rec = {"Source": row.get("pdf", "N/A"), "Full Paragraph Text": text}
        model_status = {}

        for tag, mid in MODELS.items():
            caller = (
                analyze_claude if tag.startswith("claude")
                else analyze_gemini if tag.startswith("gemini")
                else analyze_openai
            )

            print(f"[{idx+1}/{total}] {tag} … ", end="", flush=True)
            t0 = time.time()
            parsed = None
            raw_response = ""

            for attempt in range(max_retries):
                try:
                    raw_response = caller(mid, text)
                    parsed = parse_json(raw_response)
                    if not parsed:
                        raise ValueError("Parsed JSON is empty")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt+1} failed. Retrying... ", end="")
                        time.sleep(1)
                    else:
                        print(f"\nERROR ({e})")
                        if raw_response:
                           print(f"\n==== Raw FAILED response from {tag} ====\n{raw_response}\n")
                        parsed = None

            if not parsed:
                error_cols = ["Interaction Present", "Confidence", "Entities Summary",
                              "Loop Nature", "Relationship Summary",
                              "Scales Crossed", "Scale Chains", "Final Scale", "Reasoning"]
                for col in error_cols:
                    rec[f"{tag}: {col}"] = "Error: Failed to parse"
                model_status[tag] = "Error"
                continue

            present = parsed.get("interaction_present", "No")
            conf = parsed.get("confidence")
            final_scale = parsed.get("final_scale", "N/A")
            reasoning = parsed.get("reasoning", "N/A")
            
            entities_summary = "N/A"
            loop_nature, rel_summary = "N/A", "N/A"
            scales_crossed, scale_chains_formatted = "N/A", "N/A"
            
            if present == "Yes":
                entities = parsed.get("entities", [])
                relationship = parsed.get("relationship", {})
                scale_chains = parsed.get("scale_chains", [])

                loop_nature = relationship.get("loop_nature", "N/A")
                rel_summary = relationship.get("summary", "N/A")
                entities_summary = format_entities_pretty(entities)
                scales_crossed = ", ".join(map(str, scale_chains[0].get("scales_crossed", []))) if scale_chains else "N/A"
                scale_chains_formatted = format_scale_chains(scale_chains, loop_nature)

            rec.update({
                f"{tag}: Interaction Present": present,
                f"{tag}: Confidence": conf,
                f"{tag}: Entities Summary": entities_summary, 
                f"{tag}: Loop Nature": loop_nature,
                f"{tag}: Relationship Summary": rel_summary,
                f"{tag}: Scales Crossed": scales_crossed,
                f"{tag}: Scale Chains": scale_chains_formatted,
                f"{tag}: Final Scale": final_scale,
                f"{tag}: Reasoning": reasoning
            })
            model_status[tag] = present
            print(f"{present} ({time.time()-t0:.1f}s)")

        # MODIFICATION START: Replaced and enhanced the disagreement logic.
        if any(v == "Yes" for v in model_status.values()):
            # 1. General disagreement on whether an interaction is present at all
            valid_statuses = {v for v in model_status.values() if v != "Error"}
            rec["Disagreement"] = "Yes" if len(valid_statuses) > 1 else "No"
            
            # 2. Specific disagreement on the 'Final Scale'
            # Collect final scales only from models that found an interaction
            final_scales_found = {
                tag: rec.get(f"{tag}: Final Scale", "N/A")
                for tag in MODELS.keys()
                if rec.get(f"{tag}: Interaction Present") == "Yes"
            }
            
            # Check for disagreement among the found scales and create a summary
            if final_scales_found:
                unique_scales = set(final_scales_found.values())
                rec["Final Scale Disagreement"] = "Yes" if len(unique_scales) > 1 else "No"
                rec["Final Scale Summary"] = " || ".join(
                    [f"{tag}: {scale}" for tag, scale in final_scales_found.items()]
                )
            else:
                # This case is unlikely if we're inside the 'any(v=="Yes")' block, but it's safe to handle
                rec["Final Scale Disagreement"] = "No"
                rec["Final Scale Summary"] = "N/A"

            results.append(rec)
        # MODIFICATION END

    if not results:
        print("No interactions detected.")
        return

    out_df = pd.DataFrame(results)
    
    # MODIFICATION: Added the new disagreement columns to the final list
    all_cols = ["Source", "Full Paragraph Text"]
    base_cols = ["Interaction Present", "Confidence", "Entities Summary", 
                 "Loop Nature", "Relationship Summary",
                 "Scales Crossed", "Scale Chains", "Final Scale", "Reasoning"]
                 
    for tag in MODELS.keys():
        for col in base_cols:
            all_cols.append(f"{tag}: {col}")
            
    # Add the summary/disagreement columns at the end
    all_cols.extend(["Disagreement", "Final Scale Disagreement", "Final Scale Summary"])

    # Reorder DataFrame to ensure consistent column layout
    out_df = out_df.reindex(columns=all_cols)

    print(f"\nWriting {len(out_df)} rows to {args.output}")
    out_df.to_csv(args.output, index=False)
    
if __name__ == "__main__":
    main()
