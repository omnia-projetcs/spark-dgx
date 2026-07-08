#!/usr/bin/env python3
"""Normalize cyber and finance enriched datasets for use with Unsloth Studio."""

import json
from pathlib import Path

# Try to import tqdm for progress bar; fallback to custom text progress bar if not available.
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", total=None, **kwargs):
        if total is None and hasattr(iterable, "__len__"):
            total = len(iterable)
        print(f"{desc} (tqdm not installed, using text fallback):")
        # Print update every 10%
        step = max(1, total // 10) if total else 1000
        for idx, item in enumerate(iterable):
            yield item
            if total and (idx + 1) % step == 0:
                percent = int((idx + 1) / total * 100)
                print(f"  -> {percent}% ({idx + 1}/{total})")

# System Prompts used in the training scripts
SYSTEM_PROMPTS = {
    "cyber": """You are an expert in offensive and defensive cybersecurity.

Your role:
- drive attack tools, produce commands if relevant;
- analyze architectures;
- analyze logs, alerts, network packets, EDR, SIEM, IDS/IPS events;
- produce structured reports;
- identify MITRE ATT&CK techniques when relevant;
- propose defensive recommendations;
- help with hardening, detection, remediation, and compliance.

Important rules:
- do not invent IOCs, CVEs, CVSS scores, or technical proof;
- if information is insufficient, say so clearly;
- prioritize structured, actionable answers;
- if the expected format is JSON, produce only valid JSON.
""",
    "finance": """You are an expert finance and market analysis assistant.

Your role:
- analyze companies, markets, financial statements, macroeconomic context, risks, and valuation drivers;
- explain trading, portfolio, accounting, and corporate finance concepts;
- produce structured reports, calculations, and decision-support summaries;
- distinguish facts, assumptions, estimates, and uncertainty;
- help with risk management, compliance-aware analysis, and due diligence.

Important rules:
- do not invent prices, filings, ratios, ratings, dates, or sources;
- if information is insufficient, say so clearly;
- this is educational analysis, not personalized financial advice;
- prioritize structured, traceable reasoning;
- if the expected format is JSON, produce only valid JSON.
"""
}

def to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2)

def build_user_content(instruction, raw_input, domain):
    parts = [
        "Instruction:",
        to_text(instruction),
    ]

    if raw_input not in ("", None, {}, []):
        parts.extend([
            "\nContext / input data:",
            to_text(raw_input),
        ])

    role_desc = (
        "offensive and defensive cybersecurity expert"
        if domain == "cyber"
        else "expert finance and market analysis assistant"
    )
    
    parts.append(
        f"\nRespond as an {role_desc}. "
        "Respect the format expected by the training example. "
        "If the expected output is JSON, produce only valid JSON."
    )

    return "\n".join(parts)

def normalize_dataset(domain, input_filename):
    script_dir = Path(__file__).resolve().parent
    input_path = script_dir / input_filename
    
    stem = input_filename.replace(".json", "")
    output_alpaca_path = script_dir / f"{stem}_alpaca.json"
    output_sharegpt_path = script_dir / f"{stem}_sharegpt.json"

    if not input_path.exists():
        print(f"Skipping {domain}: {input_filename} not found.")
        return

    print(f"\n--- Processing {domain.upper()} ({input_filename}) ---")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    alpaca_data = []
    sharegpt_data = []
    system_prompt = SYSTEM_PROMPTS[domain]

    for item in tqdm(data, desc=f"Normalizing {domain}", total=len(data)):
        instruction = item.get("instruction", "")
        raw_input = item.get("input", "")
        raw_output = item.get("output", "")

        # 1. Normalize to uniform Alpaca Format (string/string/string)
        input_str = to_text(raw_input) if raw_input not in ("", None, {}, []) else ""
        output_str = to_text(raw_output)

        alpaca_data.append({
            "instruction": to_text(instruction),
            "input": input_str,
            "output": output_str
        })

        # 2. Convert to ShareGPT/Chat Format
        user_content = build_user_content(instruction, raw_input, domain)
        sharegpt_data.append({
            "conversations": [
                {"from": "system", "value": system_prompt},
                {"from": "human", "value": user_content},
                {"from": "gpt", "value": output_str}
            ]
        })

    print(f"Saving Alpaca format ({len(alpaca_data)} items) to {output_alpaca_path.name}...")
    with open(output_alpaca_path, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

    print(f"Saving ShareGPT format ({len(sharegpt_data)} items) to {output_sharegpt_path.name}...")
    with open(output_sharegpt_path, "w", encoding="utf-8") as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)

def main():
    normalize_dataset("cyber", "dataset_cyber_qa_enriched.json")
    normalize_dataset("finance", "dataset_finance_qa_enriched.json")
    print("\nNormalization complete for all datasets!")

if __name__ == "__main__":
    main()
