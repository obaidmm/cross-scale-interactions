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
    "claude4":   "claude-sonnet-4-20250514",
    "gpt5":      "gpt-5",
    "4o":        "gpt-4o",
    "gemini2.5": "gemini-2.5-flash",
}


# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------

CLAUDE_DETECTOR_PROMPT = r"""
<task>
You are a meticulous research analyst identifying **spatial (geographical) cross-scale interactions**. 
A valid interaction exists only when an actor at one spatial scale explicitly and causally influences, 
or is influenced by, an actor at another spatial scale. 
Output **pairwise links only**. 
If the paragraph implies A -> B -> C, output two links: A -> B and B -> C. 
Do NOT classify pattern types.
</task>

<spatial_scales>
  <scale name="Local / Site-specific">
    <description>Specific places like neighborhoods, cities, or facilities.</description>
    <examples>A zero-waste policy in a school district; a pollution issue in one town.</examples>
  </scale>
  <scale name="Sub-national / Regional">
    <description>Administrative or functional regions within a country that are larger than a specific local area.</description>
    <examples>Ontario’s electricity grid; California climate law.</examples>
  </scale>
  <scale name="National">
    <description>Entire country-level policies, actions, or discourses.</description>
    <examples>Canada’s carbon pricing; Korea’s industrial development plans.</examples>
  </scale>
  <scale name="Transnational / Cross-border">
    <description>Multi-country regional initiatives or issues that cross borders. These can also include planetary considerations.</description>
    <examples>ASEAN economic zones; EU environmental directives; cross-border smog issues; planetary boundaries; global tariffs.</examples>
  </scale>
</spatial_scales>

<protocol>
  1) Identify concrete actors and actions with spatial relevance.
  2) Map each actor to exactly one of the above scales. If unmappable, exclude it.
  3) Keep only explicit cross-scale causal links; correlation or co-occurrence is insufficient.
  4) Build pairwise links with exactly two different scales and one arrow: "Scale X -> Scale Y".
     - For A -> B -> C, output A -> B and B -> C separately. 
     - Do not output A -> C unless explicitly stated.
     - No same-scale links. No duplicates.
  5) Minimize distinct scales (≤3 if clearly necessary).
  6) Ground each link in explicit phrasing from the paragraph.
</protocol>

<output_instructions>
Return ONE valid JSON object with keys:

{
  "spatial_interaction_present": "Yes|No",
  "confidence": 0.0,
  "scales_reasoning": {
    "detected_scales": ["..."],
    "scale_explanations": {"Scale": "brief justification from the text"}
  },
  "links": [
    {"from_scale":"Scale A","to_scale":"Scale B","path":"Scale A -> Scale B","reason":"Short justification tied to explicit wording"}
  ],
  "final-link": "Scale X -> Scale Y" OR "Scale X -> Scale Y, Scale Y -> Scale Z" OR "N/A",
  "summary": "One sentence (<=200 chars) starting with bracketed unique scales, e.g., \"[Local / Site-specific, National] ...\"",
  "rationale": "2 sentences explaining your links and any scale limitation"
}

If "spatial_interaction_present" is "No":
- "detected_scales": []
- "scale_explanations": {}
- "links": []
- "final-link": "N/A"
- "summary": "N/A"
- "rationale": brief reason why no cross-scale link exists.
</output_instructions>
"""


OPENAI_DETECTOR_PROMPT = r"""
/*
SPATIAL TASK (pairwise only, no pattern types)
You are a meticulous research analyst, identify explicit spatial cross-scale interactions in the paragraph.

SPATIAL SCALE DEFINITIONS
• Local / Site-specific
  - Specific places like neighborhoods, cities, or facilities.
  - Examples: A zero-waste policy in a school district; a pollution issue in one town.
• Sub-national / Regional
  - Administrative or functional regions within a country larger than a specific local area.
  - Examples: Ontario’s electricity grid; California climate law.
• National
  - Entire country-level policies, actions, or discourses.
  - Examples: Canada’s carbon pricing; Korea’s industrial development plans.
• Transnational / Cross-border
  - Multi-country initiatives or issues crossing borders, including planetary considerations.
  - Examples: ASEAN economic zones; EU environmental directives; cross-border smog; planetary boundaries; global tariffs.

RULES
1) Extract concrete actors and actions.
2) Map each actor to one spatial scale only. If unmappable, exclude.
3) Keep only explicit cross-scale causal links (not correlation or vague co-occurrence).
4) Each link = exactly two different scales with one arrow "Scale X -> Scale Y".
   - For A -> B -> C, output A -> B and B -> C separately.
   - Do not output A -> C unless explicitly stated.
   - No duplicates. No same-scale links.
5) Keep distinct scales minimal (≤3 if necessary).
6) All links must be traceable to explicit phrasing.

OUTPUT (exact JSON object):
{
  "spatial_interaction_present": "Yes|No",
  "confidence": 0.0,
  "scales_reasoning": {
    "detected_scales": ["..."],
    "scale_explanations": {"Scale": "why ..."}
  },
  "links": [
    {"from_scale": "Scale A","to_scale":"Scale B","path":"Scale A -> Scale B","reason":"Short justification tied to explicit wording"}
  ],
  "final-link": "Scale X -> Scale Y or 'Scale X -> Scale Y, Scale Y -> Scale Z' or N/A",
  "summary": "One sentence (<=200 chars) starting with bracketed unique scales",
  "rationale": "2 sentences explaining the links and any scale limitation"
}

If "spatial_interaction_present" = "No":
- detected_scales = []
- scale_explanations = {}
- links = []
- final-link = "N/A"
- summary = "N/A"
- rationale = brief reason
*/
"""

GEMINI_DETECTOR_PROMPT = r"""
TASK: Identify explicit spatial cross-scale interactions in the paragraph. 
Use the provided scales. Output pairwise links only. Do NOT classify pattern types.

SPATIAL SCALE DEFINITIONS
- Local / Site-specific
  Specific places like neighborhoods, cities, or facilities.
  Examples: zero-waste policy in a school district; pollution issue in one town.
- Sub-national / Regional
  Regions within a country larger than a local area (states, provinces, grids).
  Examples: Ontario’s electricity grid; California climate law.
- National
  Country-level policies, actions, or discourses.
  Examples: Canada’s carbon pricing; Korea’s industrial development plans.
- Transnational / Cross-border
  Multi-country initiatives or cross-border issues, including planetary considerations.
  Examples: ASEAN zones; EU directives; cross-border smog; planetary boundaries; global tariffs.

PROTOCOL
1) Identify concrete actors and their actions.
2) Map actors to exactly one spatial scale.
3) Keep only explicit cross-scale causal links.
4) Each link: two different scales, one arrow "Scale X -> Scale Y".
   - For A -> B -> C, output A -> B and B -> C only.
   - No A -> C unless explicit. No same-scale. No duplicates.
5) Max 3 distinct scales. All links must match explicit text. The EXACT spatial scale defitions should be used in the final-link.

OUTPUT (single JSON object):
{
  "spatial_interaction_present": "Yes|No",
  "confidence": 0.0,
  "scales_reasoning": {
    "detected_scales": ["..."],
    "scale_explanations": {"Scale": "why ..."}
  },
  "links": [
    {"from_scale":"Scale A","to_scale":"Scale B","path":"Scale A -> Scale B","reason":"Short, explicit justification"}
  ],
  "final-link": "Scale X -> Scale Y or 'Scale X -> Scale Y, Scale Y -> Scale Z' or N/A",
  "summary": "One sentence (<=200 chars) starting with bracketed unique scales",
  "rationale": "2 sentences explaining links and any scale limitation"
}

If "spatial_interaction_present" = "No":
- scales = [empty]
- scale_explanations = { empty }
- detected_scales = []
- scale_explanations = { empty }
- links = []
- final-link = "N/A"
- summary = "N/A"
- rationale = brief reason
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


def analyze_openai(model: str, text: str) -> str:
    """
    Call an OpenAI model.
    Uses the new 'Responses API' for GPT-5 models based on provided documentation,
    and the standard 'Chat Completions API' for other models.
    """
    if model.startswith("gpt-5"):
        print("(Using new Responses API for GPT-5)...", end=" ", flush=True)
        full_input = f"{OPENAI_DETECTOR_PROMPT}\n\n---\n\n{text.strip()}"
        
        try:
            resp = client.responses.create(
                model=model,
                input=full_input,
                #reasoning={"effort": "low"},   # For faster, more direct responses
                text={"verbosity": "low"},     # For concise output, ideal for JSON
            )
            return resp.output_text or ""
        except Exception as e:
            print(f"\nError calling the fictional Responses API for {model}: {e}")
            return f'{{"error": "API call failed for {model}: {str(e)}"}}'

    else:
        print("(Using standard Chat Completions API)...", end=" ", flush=True)
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=800,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": OPENAI_DETECTOR_PROMPT},
                    {"role": "user", "content": text.strip()},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"\nError calling Chat Completions API for model {model}: {e}")
            return f'{{"error": "API call failed for {model}: {str(e)}"}}'


def analyze_gemini(model, paragraph):
    """Call Gemini with the spatial prompt."""
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
    """Parse a JSON object from a model reply.
    """
    if block is None:
        return None
    if isinstance(block, (dict, list)):
        return block
    s = block.strip()
    
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Fenced block fallback
    m = re.search(r'```(?:json)?\s*({.*?})\s*```', s, re.DOTALL)
    if not m:
        start, end = s.find('{'), s.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return None
        js = s[start:end + 1]
    else:
        js = m.group(1)

    try:
        return json.loads(js)
    except json.JSONDecodeError:
        # light cleanups for trailing commas / unquoted keys
        try:
            js = re.sub(r',\s*([}\]])', r'\1', js)
            js = re.sub(r'([{,])\s*([A-Za-z0-9_]+)\s*:', r'\1"\2":', js)
            return json.loads(js)
        except json.JSONDecodeError:
            return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Detect spatial interactions.")
    parser.add_argument("-i", "--input", default="paragraphs_comparison.json", help="Input JSON file with paragraphs.")
    parser.add_argument("-o", "--output_prefix", default="spatial_results", help="Output prefix (no extension).")
    parser.add_argument("-n", "--limit", type=int, default=None, help="Limit number of paragraphs to process.")
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

    raw_jsonl_path     = f"{args.output_prefix}_raw.jsonl"
    full_json_path     = f"{args.output_prefix}_full.json"
    align_json_path    = f"{args.output_prefix}_align.json"
    disagree_json_path = f"{args.output_prefix}_disagree.json"

    full_csv_path      = f"{args.output_prefix}_full.csv"
    align_csv_path     = f"{args.output_prefix}_align.csv"
    disagree_csv_path  = f"{args.output_prefix}_disagree.csv"

    rows = []
    total = len(paragraphs)
    print(f"\nStarting spatial interaction scan on {total} paragraph(s)...\n")

    with open(raw_jsonl_path, "w", encoding="utf-8") as raw_fp:
        for idx, p in enumerate(paragraphs):
            text = p["paragraph"]
            source = p["pdf"]

            rec = {
                "A: Paragraph Index": idx,
                "Source": source,
                "Full Paragraph Text": text,
            }

            model_status = {}
            per_model_json = {}

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
                        print(f"\n[gpt5 raw head] {str(response)[:300]}\n")
                        raise ValueError("Invalid JSON")

                    present = parsed.get("spatial_interaction_present", "No")
                    conf = parsed.get("confidence")
                    sr = parsed.get("scales_reasoning", {}) or {}
                    detected_scales = list(sr.get("detected_scales", []) or [])
                    explanations = sr.get("scale_explanations", {}) or {}
                    links = parsed.get("links", []) or []
                    final_link = parsed.get("final-link", "N/A")
                    summary = parsed.get("summary") or "N/A"
                    rationale = parsed.get("rationale") or "N/A"

                    scales_str = ", ".join(detected_scales) if detected_scales else "N/A"
                    link_paths_list = [ (l.get("path") or "").strip() for l in links if l.get("path") ]
                    links_path = " | ".join(link_paths_list) if link_paths_list else "N/A"

                    model_status[tag] = present if present in ("Yes", "No") else "Error"
                    rec.update({
                        f"{tag}: Interaction Present": present,
                        f"{tag}: Confidence": conf if conf is not None else "N/A",
                        f"{tag}: Scales": scales_str,
                        f"{tag}: Links": links_path,
                        f"{tag}: Final Link": final_link if final_link else "N/A",
                        f"{tag}: Scale Explanations": "; ".join([f"{k}: {v}" for k, v in explanations.items()]) if explanations else "N/A",
                        f"{tag}: Summary": summary,
                        f"{tag}: Rationale": rationale,
                    })

                    per_model_json[tag] = parsed
                    print(f"Result: {present} (took {elapsed:.1f}s)")

                except Exception as e:
                    model_status[tag] = "Error"
                    per_model_json[tag] = {"error": str(e)}
                    for col in [
                        "Interaction Present", "Confidence", "Scales", "Links",
                        "Final Link", "Scale Explanations", "Summary", "Rationale"
                    ]:
                        rec[f"{tag}: {col}"] = "Error"
                    print(f"ERROR: {e}")

            # raw JSONL line for auditing
            raw_fp.write(json.dumps({
                "paragraph_index": idx,
                "source": source,
                "paragraph": text,
                "models": per_model_json
            }, ensure_ascii=False) + "\n")

            valid = [s for s in model_status.values() if s in ("Yes", "No")]
            has_error = any(s == "Error" for s in model_status.values())

            consensus = "Mixed/Error" 
            all_align = "No"

            if valid:
                # If ALL models that ran without error agree
                if len(set(valid)) == 1:
                    consensus = valid[0] # The consensus is "Yes" or "No"
                    # Only mark "All Models Align" as "Yes" if there were NO errors
                    if not has_error and len(valid) == len(MODELS):
                        all_align = "Yes"
                # only if there are valid results but they disagree 
                else:
                    consensus = "Mixed"
            
            rec["All Models Align"] = all_align
            rec["B: AI: Cross Scale (Consensus)"] = consensus
            rec["C: Human Agreement with B (Agree/Disagree/Notes)"] = ""
            rec["Disagreement Type"] = (
                "Unanimous" if rec["All Models Align"] == "Yes"
                else ("Split Yes/No" if set(valid) == {"Yes", "No"} else "Error/Partial")
            )

            rows.append(rec)

    print(f"\nSaved raw JSONL to {raw_jsonl_path}")

    def _save_json(path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} rows to {path}")

    align_rows    = [r for r in rows if r["All Models Align"] == "Yes"]
    disagree_rows = [r for r in rows if r["All Models Align"] == "No"]

    _save_json(full_json_path, rows)
    _save_json(align_json_path, align_rows)
    _save_json(disagree_json_path, disagree_rows)

    def _reorder(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        ordered = [
            "A: Paragraph Index",
            "B: AI: Cross Scale (Consensus)",
            "C: Human Agreement with B (Agree/Disagree/Notes)",
            "All Models Align",
            "Disagreement Type",
            "Source",
            "Full Paragraph Text",
        ]
        per_model_suffixes = [
            "Interaction Present", "Confidence", "Scales", "Links", "Final Link",
            "Scale Explanations", "Summary", "Rationale"
        ]
        for tag in MODELS.keys():
            for suf in per_model_suffixes:
                ordered.append(f"{tag}: {suf}")
        keep = [c for c in ordered if c in df.columns]
        return df[keep + [c for c in df.columns if c not in keep]]

    df_full = _reorder(pd.DataFrame(rows))
    df_align = _reorder(pd.DataFrame(align_rows))
    df_disagree = _reorder(pd.DataFrame(disagree_rows))

    df_full.to_csv(full_csv_path, index=False)
    df_align.to_csv(align_csv_path, index=False)
    df_disagree.to_csv(disagree_csv_path, index=False)

    print(f"Saved {len(df_full)} rows to {full_csv_path}")
    print(f"Saved {len(df_align)} rows to {align_csv_path}")
    print(f"Saved {len(df_disagree)} rows to {disagree_csv_path}")

if __name__ == "__main__":
    main()