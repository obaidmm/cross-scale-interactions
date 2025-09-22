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
import google.generativeai as genai

# ---------------------------------------------------------------------------
# API setup
# ---------------------------------------------------------------------------
claude        = anthropic.Client(api_key=config.ANTHROPIC_API_KEY)
client        = OpenAI(api_key=config.OPENAI_API_KEY)
# client_gemini = genai.Client(api_key=config.GOOGLE_API_KEY)
genai.configure(api_key=config.GOOGLE_API_KEY)


MODELS = {
    "claude4":   "claude-sonnet-4-20250514",
    "gpt5":      "gpt-5",
    "4o":        "gpt-4o",
    "gemini2.5": "gemini-2.5-flash",
}


# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------

PROMPT = r"""
You are a research assistant. Identify interactions within each scale dimension in the paragraph and return JSON.

Dimensions:
- Structural: individuals, organizations, inter-organizations, environment & society
- Temporal: short-term, mid-term, long-term
- Spatial: local, regional (within nations), national, transnational

Interaction types:
- None
- Causal only
- Reinforcing feedback
- Balancing feedback

Rules:
- Only use what is explicitly stated in the text.
- Capture interactions **within one dimension at a time** (structural-to-structural, temporal-to-temporal, spatial-to-spatial).
- Each interaction = from_level, to_level, type, and short evidence phrase.
- If no interactions exist for a dimension, return an empty array.

Output JSON format:
{
  "topic": "few words",
  "structural_interactions": [
    {"from_level": "...", "to_level": "...", "type": "...", "evidence": "..."}
  ],
  "temporal_interactions": [
    {"from_level": "...", "to_level": "...", "type": "...", "evidence": "..."}
  ],
  "spatial_interactions": [
    {"from_level": "...", "to_level": "...", "type": "...", "evidence": "..."}
  ],
  "notes": "Short clarification or 'No interactions detected'."
}
"""




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def analyze_claude(model_id: str, paragraph: str) -> str:
    """Call Claude with the temporal prompt."""
    resp = claude.messages.create(
        model=model_id,
        system=PROMPT,
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
        full_input = f"{PROMPT}\n\n---\n\n{text.strip()}"
        
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
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": text.strip()},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"\nError calling Chat Completions API for model {model}: {e}")
            return f'{{"error": "API call failed for {model}: {str(e)}"}}'

def analyze_gemini(model_name, paragraph):
    """Call a Gemini model with specific generation configuration."""
    try:
        model = genai.GenerativeModel(model_name)
        
        generation_config = {
            "temperature": 0,
            "max_output_tokens": 8192, # Use the max to avoid errors
            "response_mime_type": "application/json",
        }
        full_prompt = PROMPT + "\n" + paragraph.strip()
        
        res = model.generate_content(
            contents=[full_prompt],
            generation_config=generation_config
        )
   
        return res.text

    except Exception as e:
        print(f"\nCaught an exception in analyze_gemini: {e}")
        try:
            # This provides the specific reason (e.g., SAFETY, MAX_TOKENS)
            feedback = res.prompt_feedback
            finish_reason = res.candidates[0].finish_reason
            return f'{{"error": "API call failed", "feedback": "{feedback}", "finish_reason": "{finish_reason.name}"}}'
        except:
            return f'{{"error": "API call failed with an unrecoverable error: {str(e)}"}}'
        

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
    parser = argparse.ArgumentParser(description="Detect scale interactions.")
    parser.add_argument(
        "-i", "--input", default="paragraphs.json", help="Input JSON file with paragraphs."
    )
    parser.add_argument(
        "-o", "--output_prefix", default="results", help="Output prefix (no extension)."
    )
    parser.add_argument(
        "-n", "--limit", type=int, default=None, help="Limit number of paragraphs."
    )
    args = parser.parse_args()

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            paragraphs_data = json.load(f)
    except Exception as e:
        print(f"Failed to load input: {e}")
        return

    paragraphs = paragraphs_data[: args.limit] if args.limit else paragraphs_data
    if not paragraphs:
        print("No data found.")
        return

    rows = []
    total = len(paragraphs)
    print(f"\nStarting scan on {total} paragraph(s)...\n")

    for idx, p in enumerate(paragraphs):
        text = p["paragraph"]
        source = p.get("pdf", "N/A")

        rec = {"Index": idx, "Source": source, "Paragraph": text}
        per_model_json = {}

        print(f"\n--- Paragraph {idx+1}/{total} ---")

        for tag, mid in MODELS.items():
            try:
                print(f"Querying {tag}...", end=" ", flush=True)
                t0 = time.time()
                if tag.startswith("claude"):
                    response = analyze_claude(mid, text)
                elif tag.startswith("gemini"):
                    response = analyze_gemini(mid, text)
                else:
                    response = analyze_openai(mid, text)
                parsed = parse_json(response)
                elapsed = time.time() - t0

                if not parsed:
                    raise ValueError("Invalid JSON")

                topic = parsed.get("topic", "N/A")
                struct_links = parsed.get("structural_interactions", [])
                temp_links = parsed.get("temporal_interactions", [])
                spat_links = parsed.get("spatial_interactions", [])
                notes = parsed.get("notes", "")

                def _fmt_links(arr):
                    return " | ".join(
                        f"{l.get('from_level')} -> {l.get('to_level')} [{l.get('type')}] :: {l.get('evidence')}"
                        for l in arr
                    ) if arr else "N/A"

                rec.update({
                    f"{tag}: Topic": topic,
                    f"{tag}: Structural Links": _fmt_links(struct_links),
                    f"{tag}: Temporal Links": _fmt_links(temp_links),
                    f"{tag}: Spatial Links": _fmt_links(spat_links),
                    f"{tag}: Notes": notes,
                })
                per_model_json[tag] = parsed
                print(f"done ({elapsed:.1f}s)")
            except Exception as e:
                rec.update({
                    f"{tag}: Topic": "Error",
                    f"{tag}: Structural Links": "Error",
                    f"{tag}: Temporal Links": "Error",
                    f"{tag}: Spatial Links": "Error",
                    f"{tag}: Notes": str(e),
                })
                print(f"ERROR: {e}")
        rows.append(rec)

    # Save outputs
    out_json = f"{args.output_prefix}.json"
    out_csv = f"{args.output_prefix}.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(rows)} rows to {out_json}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()