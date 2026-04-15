#!/usr/bin/env python3
"""
generate_verbs_gemini.py  (fixed version)
==========================================
Uses gemini-2.0-flash-lite  — higher free quota, less rate limiting
Free tier: 30 requests/min, 1500/day
"""

import os, sys, json, time, argparse

SOURCE_FILE   = "verbs_source.json"
OUTPUT_FILE   = "verbs_generated.json"
MODEL         = "gemini-3-flash-preview"   # <-- higher free quota than flash
BATCH_SIZE    = 15
DELAY_SECONDS = 2   # 2.5s between batches = ~24 req/min, safely under 30 limit

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--key",   default=os.environ.get("GEMINI_API_KEY",""))
    p.add_argument("--model", default=MODEL)
    p.add_argument("--batch", type=int,   default=BATCH_SIZE)
    p.add_argument("--delay", type=float, default=DELAY_SECONDS)
    p.add_argument("--limit", type=int,   default=0)
    return p.parse_args()

PROMPT = '''Generate all 4 English tenses and their German translations for each verb below.
Return ONLY a valid JSON array — no markdown, no code fences, nothing else.

Each element:
{{
  "verb": "<exactly as given>",
  "present":  {{"en_verb":"...","en_sentence":"...","de_verb":"...","de_sentence":"..."}},
  "past":     {{"en_verb":"...","en_sentence":"...","de_verb":"...","de_sentence":"..."}},
  "perfect":  {{"en_verb":"...","en_sentence":"...","de_verb":"...","de_sentence":"..."}},
  "future":   {{"en_verb":"...","en_sentence":"...","de_verb":"...","de_sentence":"..."}}
}}

Rules:
- present en_sentence: use EXACTLY the provided sentence
- en_verb: conjugated form only (runs / ran / had run / will run)
- de_verb: German verb form only (läuft / lief / ist gelaufen / wird laufen)
- de_sentence: natural correct German
- perfect: "had + pp" in English, "haben/sein + Partizip II" in German
- All sentences must be natural, not robotic

Verbs:
{verb_list}'''

def call_gemini(batch, model_name, api_key):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(model_name)
    verb_list = "\n".join(
        f'- verb="{v["verb"]}", present_sentence="{v["en_present_sentence"]}"'
        for v in batch
    )
    resp = m.generate_content(PROMPT.format(verb_list=verb_list))
    text = resp.text.strip()
    # strip any accidental markdown fences
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"): p = p[4:].strip()
            if p.startswith("["): 
                text = p
                break
    return json.loads(text.strip())

def empty(verb, sentence):
    return {"verb": verb, "en_present_sentence": sentence,
            "present": {"en_verb":verb,"en_sentence":sentence,"de_verb":"","de_sentence":""},
            "past":    {"en_verb":"","en_sentence":"","de_verb":"","de_sentence":""},
            "perfect": {"en_verb":"","en_sentence":"","de_verb":"","de_sentence":""},
            "future":  {"en_verb":f"will {verb}","en_sentence":"","de_verb":"","de_sentence":""}}

def save(results):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, separators=(",",":"))

def main():
    args = parse_args()
    if not args.key:
        print("❌  No API key. Get free key: https://aistudio.google.com/apikey")
        print("    Run: python generate_verbs_gemini.py --key AIza...")
        sys.exit(1)
    try:
        import google.generativeai
    except ImportError:
        print("❌  Run: pip install google-generativeai")
        sys.exit(1)

    with open(SOURCE_FILE, encoding="utf-8") as f:
        source = json.load(f)
    print(f"📖  {len(source)} verbs loaded")

    results, done_set = [], set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            results = json.load(f)
        done_set = {r["verb"].lower() for r in results}
        print(f"⏩  Resuming — {len(done_set)} already done")

    todo = [v for v in source if v["verb"].lower() not in done_set]
    if args.limit > 0:
        todo = todo[:args.limit]
    if not todo:
        print("✅  All done!"); return

    total_batches = (len(todo) + args.batch - 1) // args.batch
    eta_min = total_batches * args.delay / 60
    print(f"🔧  Model: {args.model}")
    print(f"📦  {total_batches} batches × {args.batch} verbs  |  ~{eta_min:.1f} min  |  FREE tier")
    print(f"💾  → {OUTPUT_FILE}")
    print("─" * 55)

    t0, failed = time.time(), []

    for batch_num, i in enumerate(range(0, len(todo), args.batch)):
        batch = todo[i:i + args.batch]
        success = False

        for attempt in range(4):
            try:
                batch_results = call_gemini(batch, args.model, args.key)
                for res in batch_results:
                    src = next((v for v in batch if v["verb"].lower() == res["verb"].lower()), None)
                    if src:
                        results.append({"verb": src["verb"],
                                        "en_present_sentence": src["en_present_sentence"],
                                        **res})
                success = True
                break

            except json.JSONDecodeError as e:
                wait = 4 * (attempt + 1)
                print(f"  ⚠  JSON error (attempt {attempt+1}): {str(e)[:60]} — retry in {wait}s")
                time.sleep(wait)

            except Exception as e:
                msg = str(e)
                is_rate = any(x in msg for x in ["429","quota","RESOURCE_EXHAUSTED","rate"])
                if is_rate:
                    # exponential backoff: 30s, 60s, 120s
                    wait = 30 * (2 ** attempt)
                    print(f"  ⏸  Rate limited (attempt {attempt+1}) — waiting {wait}s...")
                    time.sleep(wait)
                else:
                    wait = 5 * (attempt + 1)
                    print(f"  ⚠  Error: {msg[:80]} — retry in {wait}s")
                    time.sleep(wait)

        if not success:
            print(f"  ✗  Giving up on batch, inserting empty placeholders")
            for v in batch:
                results.append(empty(v["verb"], v["en_present_sentence"]))
                failed.append(v["verb"])

        # progress
        done = len(results)
        pct  = done / len(source) * 100
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 1
        eta  = (len(source) - done) / rate
        vstr = ", ".join(v["verb"] for v in batch[:3]) + ("…" if len(batch)>3 else "")
        print(f"  [{done:4d}/{len(source)}] {pct:5.1f}%  ✓  {vstr}  (ETA {eta/60:.1f}min)")

        # checkpoint every 30 verbs
        if done % 30 < args.batch:
            save(results)

        if i + args.batch < len(todo):
            time.sleep(args.delay)

    save(results)
    print("─" * 55)
    print(f"✅  {len(results)} verbs done in {(time.time()-t0)/60:.1f} min")
    if failed:
        print(f"⚠   Failed: {', '.join(failed[:10])}")
    print(f"\n📦  '{OUTPUT_FILE}' ready — drag into the verb trainer app!")

if __name__ == "__main__":
    main()
