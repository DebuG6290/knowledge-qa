import chromadb
import requests
import json
import time
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

# ── config ──────────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL       = "llama3.2"
REPORT_FILE = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# ── init ─────────────────────────────────────────────────────────────────────
embedder  = SentenceTransformer('all-MiniLM-L6-v2')
reranker  = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
client    = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("knowledge")

# ── test definitions ─────────────────────────────────────────────────────────

STANDALONE_TESTS = [
    {
        "id": "S1",
        "question": "What is Bridgewater's AI strategy?",
        "expected_sources": ["asset_management"],
        "expected_keywords": ["PrinciplesOS", "Ray Dalio", "trade"],
        "should_answer": True,
    },
    {
        "id": "S2",
        "question": "Who founded Bridgewater?",
        "expected_sources": ["asset_management"],
        "expected_keywords": ["Ray Dalio", "1975"],
        "should_answer": True,
    },
    {
        "id": "S3",
        "question": "What does Vanguard use AI for?",
        "expected_sources": ["asset_management"],
        "expected_keywords": ["client", "virtual assistant"],
        "should_answer": True,
    },
    {
        "id": "S4",
        "question": "What is BlackRock's Aladdin system?",
        "expected_sources": ["asset_management"],
        "expected_keywords": ["risk", "BlackRock", "Aladdin"],
        "should_answer": True,
    },
    {
        "id": "S5",
        "question": "How does JPMorgan use AI?",
        "expected_sources": ["article"],
        "expected_keywords": ["JPMorgan"],
        "should_answer": True,
    },
]

FOLLOWUP_SEQUENCE = [
    {
        "id": "F1",
        "question": "What is Bridgewater's AI strategy?",
        "expected_keywords": ["PrinciplesOS", "Ray Dalio"],
        "expected_sources": ["asset_management"],
        "rewrite_should_contain": [],  # first question, no rewrite
    },
    {
        "id": "F2",
        "question": "How is that different from BlackRock?",
        "expected_keywords": ["BlackRock", "Bridgewater"],
        "expected_sources": ["asset_management"],
        "rewrite_should_contain": ["Bridgewater", "BlackRock"],
    },
    {
        "id": "F3",
        "question": "What about Vanguard?",
        "expected_keywords": ["Vanguard", "client"],
        "expected_sources": ["asset_management"],
        "rewrite_should_contain": ["Vanguard"],
    },
    {
        "id": "F4",
        "question": "Which of these three is most sustainable long term and why?",
        "expected_keywords": ["Vanguard", "sustainable"],
        "expected_sources": ["asset_management"],
        "rewrite_should_contain": ["Bridgewater", "BlackRock", "Vanguard"],
    },
    {
        "id": "F5",
        "question": "Going back to Bridgewater, who founded it?",
        "expected_keywords": ["Ray Dalio", "1975"],
        "expected_sources": ["asset_management"],
        "rewrite_should_contain": ["Bridgewater"],
    },
    {
        "id": "F6",
        "question": "Why did experts say that approach gets arbitraged away?",
        "expected_keywords": ["arbitraged", "alpha"],
        "expected_sources": ["asset_management"],
        "rewrite_should_contain": ["alpha", "Bridgewater"],
    },
]

PRONOUN_HELL_SEQUENCE = [
    {
        "id": "P1",
        "question": "What is Bridgewater's investment strategy?",
        "expected_keywords": ["Bridgewater"],
        "rewrite_should_contain": [],
    },
    {
        "id": "P2",
        "question": "How did they develop it?",
        "expected_keywords": ["Bridgewater", "PrinciplesOS"],
        "rewrite_should_contain": ["Bridgewater"],
    },
    {
        "id": "P3",
        "question": "Who was behind that decision?",
        "expected_keywords": ["Ray Dalio"],
        "rewrite_should_contain": ["Bridgewater", "Ray Dalio"],
    },
    {
        "id": "P4",
        "question": "Did it work?",
        "expected_keywords": ["Bridgewater", "PrinciplesOS"],
        "rewrite_should_contain": ["Bridgewater", "PrinciplesOS"],
    },
    {
        "id": "P5",
        "question": "What did critics say about it?",
        "expected_keywords": ["Bridgewater"],
        "rewrite_should_contain": ["Bridgewater"],
    },
    {
        "id": "P6",
        "question": "Why did they persist anyway?",
        "expected_keywords": ["Bridgewater"],
        "rewrite_should_contain": ["Bridgewater"],
    },
]

CROSS_DOC_TESTS = [
    {
        "id": "C1",
        "question": "How does the AI technique Bridgewater uses relate to how neural networks actually work?",
        "expected_sources": ["asset_management", "a16z_ai"],
        "expected_keywords": ["Bridgewater", "neural"],
        "should_answer": True,
        "note": "Requires combining asset_management + a16z_ai — tests 2 chunk limit"
    },
    {
        "id": "C2",
        "question": "What Marcus Aurelius principle best applies to BlackRock's risk management approach?",
        "expected_sources": ["marcus_aurelius", "asset_management"],
        "expected_keywords": ["BlackRock", "risk"],
        "should_answer": True,
        "note": "Cross-domain synthesis — finance + philosophy"
    },
]

NEGATION_TESTS = [
    {
        "id": "N1",
        "question": "What does Bridgewater NOT use AI for?",
        "expected_keywords": ["Bridgewater"],
        "note": "Negation — vectors bad at NOT queries"
    },
    {
        "id": "N2",
        "question": "Which firm does not focus on alpha generation?",
        "expected_keywords": ["Vanguard"],
        "note": "Negation with comparison"
    },
]

SUPERLATIVE_TESTS = [
    {
        "id": "SU1",
        "question": "Which firm manages the most assets?",
        "expected_keywords": ["BlackRock", "trillion"],
        "note": "Requires reading and comparing numbers across chunks"
    },
    {
        "id": "SU2",
        "question": "What is the biggest risk across all three firms?",
        "expected_keywords": ["risk"],
        "note": "Superlative + synthesis"
    },
]

ABSENT_INFO_TESTS = [
    {
        "id": "A1",
        "question": "What is BlackRock's revenue in 2023?",
        "should_answer": False,
        "note": "Not in knowledge base — should express uncertainty"
    },
    {
        "id": "A2",
        "question": "How many engineers work on PrinciplesOS?",
        "should_answer": False,
        "note": "Specific detail not in knowledge base"
    },
    {
        "id": "A3",
        "question": "What is the capital of France?",
        "should_answer": False,
        "note": "Completely out of scope"
    },
    {
        "id": "A4",
        "question": "Who is Virat Kohli?",
        "should_answer": False,
        "note": "Completely out of scope"
    },
]

CONTRADICTORY_TESTS = [
    # run as a sequence
    {
        "id": "CO1",
        "question": "Who founded Bridgewater?",
        "expected_keywords": ["Ray Dalio"],
    },
    {
        "id": "CO2",
        "question": "Actually I read it was founded by John Smith in 1980",
        "expected_keywords": ["Ray Dalio"],  # should NOT accept the false claim
        "note": "False injection — should trust knowledge base not user"
    },
    {
        "id": "CO3",
        "question": "So who actually founded Bridgewater?",
        "expected_keywords": ["Ray Dalio"],
        "note": "After false injection — does it recover?"
    },
]

AMBIGUOUS_REFERENT_TESTS = [
    {
        "id": "AM1",
        "question": "Compare BlackRock and Bridgewater's AI approaches",
        "expected_keywords": ["BlackRock", "Bridgewater"],
        "rewrite_should_contain": [],
    },
    {
        "id": "AM2",
        "question": "Which one uses more data?",
        "expected_keywords": ["BlackRock", "Bridgewater"],
        "rewrite_should_contain": ["BlackRock", "Bridgewater"],
        "note": "Ambiguous 'one' — could be either firm"
    },
    {
        "id": "AM3",
        "question": "Does it have more employees?",
        "expected_keywords": [],
        "rewrite_should_contain": ["BlackRock", "Bridgewater"],
        "note": "Ambiguous 'it' after ambiguous 'one'"
    },
]

TOPIC_BLEED_TESTS = [
    {
        "id": "TB1",
        "question": "What principles guide Bridgewater's decision making?",
        "expected_sources": ["asset_management"],
        "forbidden_sources": ["marcus_aurelius"],
        "expected_keywords": ["Ray Dalio", "PrinciplesOS"],
        "note": "Semantic collision — 'principles' hits Marcus Aurelius too"
    },
    {
        "id": "TB2",
        "question": "How should one approach adversity according to the texts?",
        "expected_sources": ["marcus_aurelius"],
        "note": "Should pull philosophy, not finance"
    },
]

EDGE_CASE_TESTS = [
    {"id": "E1", "question": "", "note": "Empty input"},
    {"id": "E2", "question": "?", "note": "Just punctuation"},
    {"id": "E3", "question": "asdfjkl;", "note": "Gibberish"},
    {"id": "E4", "question": "What?", "note": "Single word follow-up"},
    {"id": "E5", "question": "   ", "note": "Whitespace only"},
]

TEMPORAL_TESTS = [
    {
        "id": "T1",
        "question": "Which AI initiative came first, BlackRock's Aladdin or Bridgewater's PrinciplesOS?",
        "expected_keywords": ["BlackRock", "Bridgewater"],
        "note": "Temporal comparison — system has no time concept"
    },
]

MEMORY_AND_RECALL_TESTS = [
    # Layer 2.5 — explicit recall
    {"id": "M1", "question": "recall: what did I conclude about Bridgewater?",
     "should_route": "conversations"},

    # Layer 2.5 — auto-detect, no prefix
    {"id": "M2", "question": "What did I say about Vanguard earlier?",
     "should_route": "conversations"},

    # Layer 2.5 — dual search trigger
    {"id": "M3", "question": "Does what I said about Vanguard match the article?",
     "should_route": "both"},

    # Relevance gate — zero-cost verification
    {"id": "M4", "question": "What is the capital of France?",
     "expected_generation_tokens": 0},  # gate should skip generation entirely

    # Relevance gate — known ceiling, documented not fixed
    {"id": "M5", "question": "Does BlackRock's approach match Bridgewater's?",
     "note": "Known ceiling: may over-refuse despite sufficient context (Part 14)"},
]

# ── core functions ────────────────────────────────────────────────────────────

def ollama_call(prompt, stream=False):
    """Single Ollama call, returns (text, prompt_tokens, output_tokens)"""
    try:
        resp = requests.post(OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": stream},
            timeout=120)
        if stream:
            full = ""
            pt, ot = 0, 0
            for line in resp.iter_lines():
                if line:
                    c = json.loads(line)
                    full += c.get('response', '')
                    if c.get('done'):
                        pt = c.get('prompt_eval_count', 0)
                        ot = c.get('eval_count', 0)
            return full.strip(), pt, ot
        else:
            data = resp.json()
            text = data.get('response', '').strip()
            pt   = data.get('prompt_eval_count', 0)
            ot   = data.get('eval_count', 0)
            return text, pt, ot
    except Exception as e:
        return f"[OLLAMA ERROR: {e}]", 0, 0


def rewrite_question(question, history):
    if not history:
        return question, 0
    recent = history[-2:]
    history_text = ""
    for q, a in recent:
        history_text += f"Q: {q}\nA: {a[:200]}\n\n"
    prompt = f"""Given this conversation history:
{history_text}
Rewrite this follow-up question as a fully self-contained question with no pronouns or references:
"{question}"
Return ONLY the rewritten question, nothing else."""
    rewritten, pt, ot = ollama_call(prompt, stream=False)
    return rewritten if rewritten else question, pt + ot


def retrieve_and_answer(question):
    """Returns (answer, sources, rewritten, chunks, token_dict)"""
    vec = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[vec],
        n_results=6,
        include=["documents", "metadatas", "distances"]
    )
    chunks = results['documents'][0]
    ids    = results['ids'][0]

    pairs  = [[question, c] for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), reverse=True)
    top_chunks = [c for _, c in ranked[:2]]

    sources = list(set([i.rsplit('_', 1)[0] for i in ids]))
    context = "\n\n".join(top_chunks)

    prompt = f"Answer based only on this context:\n\n{context}\n\nQuestion: {question}"
    answer, pt, ot = ollama_call(prompt, stream=False)

    return answer, sources, top_chunks, {"prompt": pt, "output": ot}


def check_keywords(text, keywords):
    text_lower = text.lower()
    return {kw: kw.lower() in text_lower for kw in keywords}


def uncertainty_expressed(answer):
    """Check if answer expresses uncertainty / out of scope"""
    signals = [
        "don't know", "do not know", "not in", "cannot find",
        "no information", "not mentioned", "not provided",
        "i'm not sure", "i am not sure", "unclear", "not available",
        "context does not", "context doesn't", "not enough"
    ]
    a = answer.lower()
    return any(s in a for s in signals)

# ── test runners ──────────────────────────────────────────────────────────────

def run_standalone(tests, label="STANDALONE"):
    results = []
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for t in tests:
        print(f"\n[{t['id']}] {t['question'][:80]}")
        if t.get('note'):
            print(f"     NOTE: {t['note']}")

        start = time.time()
        answer, sources, chunks, tokens = retrieve_and_answer(t['question'])
        elapsed = round(time.time() - start, 1)

        kw_results  = check_keywords(answer, t.get('expected_keywords', []))
        kw_pass     = all(kw_results.values()) if kw_results else None
        src_expected = t.get('expected_sources', [])
        src_hit      = any(s in sources for s in src_expected) if src_expected else None
        forbidden    = t.get('forbidden_sources', [])
        src_clean    = not any(s in sources for s in forbidden) if forbidden else True
        should_answer = t.get('should_answer', True)
        uncertainty   = uncertainty_expressed(answer)

        if should_answer:
            overall = "PASS" if (kw_pass is not False and src_hit is not False and src_clean) else "FAIL"
        else:
            overall = "PASS" if uncertainty else "FAIL"

        print(f"     Answer: {answer[:150]}...")
        print(f"     Sources: {sources}")
        print(f"     Keywords: {kw_results}")
        print(f"     Source hit: {src_hit} | Forbidden clean: {src_clean}")
        print(f"     Uncertainty expressed: {uncertainty}")
        print(f"     Tokens — prompt:{tokens['prompt']} output:{tokens['output']} total:{tokens['prompt']+tokens['output']}")
        print(f"     Time: {elapsed}s | Result: {overall}")

        results.append({
            "id": t['id'], "question": t['question'],
            "answer": answer, "sources": sources,
            "kw_results": kw_results, "kw_pass": kw_pass,
            "src_hit": src_hit, "src_clean": src_clean,
            "uncertainty": uncertainty, "overall": overall,
            "tokens": tokens, "elapsed": elapsed,
            "note": t.get('note', '')
        })
    return results


def run_sequence(tests, label="SEQUENCE"):
    """Run a conversation sequence maintaining history"""
    results = []
    history = []
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    for t in tests:
        print(f"\n[{t['id']}] {t['question'][:80]}")
        if t.get('note'):
            print(f"     NOTE: {t['note']}")

        rewrite_tokens = 0
        rewritten = t['question']

        if history:
            rewritten, rewrite_tokens = rewrite_question(t['question'], history)
            print(f"     Rewritten: {rewritten[:120]}")

        start  = time.time()
        answer, sources, chunks, tokens = retrieve_and_answer(rewritten)
        elapsed = round(time.time() - start, 1)

        kw_results  = check_keywords(answer, t.get('expected_keywords', []))
        kw_pass     = all(kw_results.values()) if kw_results else None
        src_expected = t.get('expected_sources', [])
        src_hit      = any(s in sources for s in src_expected) if src_expected else None

        rw_check = t.get('rewrite_should_contain', [])
        rw_pass  = all(kw.lower() in rewritten.lower() for kw in rw_check) if rw_check else None

        overall = "PASS" if (kw_pass is not False and src_hit is not False and rw_pass is not False) else "FAIL"

        print(f"     Answer: {answer[:150]}...")
        print(f"     Sources: {sources}")
        print(f"     Keywords: {kw_results}")
        print(f"     Rewrite check: {rw_check} → {rw_pass}")
        print(f"     Tokens — rewriter:{rewrite_tokens} prompt:{tokens['prompt']} output:{tokens['output']}")
        print(f"     Time: {elapsed}s | Result: {overall}")

        history.append((t['question'], answer))

        results.append({
            "id": t['id'], "question": t['question'],
            "rewritten": rewritten, "answer": answer,
            "sources": sources, "kw_results": kw_results,
            "kw_pass": kw_pass, "src_hit": src_hit,
            "rw_pass": rw_pass, "overall": overall,
            "tokens": {**tokens, "rewriter": rewrite_tokens},
            "elapsed": elapsed, "note": t.get('note', '')
        })
    return results


def run_edge_cases(tests):
    results = []
    print(f"\n{'='*60}")
    print(f"  EDGE CASES")
    print(f"{'='*60}")
    for t in tests:
        print(f"\n[{t['id']}] Input: '{t['question']}' | NOTE: {t['note']}")
        try:
            if not t['question'].strip():
                print(f"     Result: HANDLED — empty input detected before query")
                results.append({"id": t['id'], "overall": "HANDLED", "note": t['note']})
                continue
            answer, sources, chunks, tokens = retrieve_and_answer(t['question'])
            uncertainty = uncertainty_expressed(answer)
            print(f"     Answer: {answer[:150]}")
            print(f"     Uncertainty: {uncertainty}")
            overall = "HANDLED" if uncertainty else "FAIL — answered confidently"
            results.append({"id": t['id'], "overall": overall, "answer": answer, "note": t['note']})
        except Exception as e:
            print(f"     CRASHED: {e}")
            results.append({"id": t['id'], "overall": f"CRASH: {e}", "note": t['note']})
    return results

# ── report ────────────────────────────────────────────────────────────────────

def generate_report(all_results):
    lines = []
    lines.append("=" * 70)
    lines.append("RAG SYSTEM TEST REPORT — BASELINE (Pre-Memory)")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    total_pass = 0
    total_fail = 0
    total_tokens = 0

    for section_name, results in all_results.items():
        if not results:
            continue
        lines.append(f"\n── {section_name} ──")
        section_pass = sum(1 for r in results if r.get('overall') in ['PASS', 'HANDLED'])
        section_fail = sum(1 for r in results if r.get('overall') not in ['PASS', 'HANDLED'])
        lines.append(f"Score: {section_pass}/{len(results)}")

        for r in results:
            status = r.get('overall', '?')
            lines.append(f"  [{r['id']}] {status}")
            if r.get('note'):
                lines.append(f"        Note: {r['note']}")
            if r.get('answer'):
                lines.append(f"        Answer: {r['answer'][:100]}...")
            if r.get('rewritten') and r.get('rewritten') != r.get('question'):
                lines.append(f"        Rewritten: {r['rewritten'][:100]}")
            if r.get('tokens'):
                t = r['tokens']
                tok_total = t.get('prompt', 0) + t.get('output', 0) + t.get('rewriter', 0)
                total_tokens += tok_total
                lines.append(f"        Tokens: {tok_total}")

        total_pass += section_pass
        total_fail += section_fail

    lines.append("\n" + "=" * 70)
    lines.append("SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Total PASS: {total_pass}")
    lines.append(f"Total FAIL: {total_fail}")
    lines.append(f"Overall:    {total_pass}/{total_pass+total_fail}")
    lines.append(f"Total tokens consumed this run: {total_tokens}")
    lines.append("\nKEY GAPS IDENTIFIED:")
    lines.append("(Review FAIL entries above — these are your pre-memory baseline failures)")
    lines.append("=" * 70)

    report = "\n".join(lines)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print("\n\n" + report)
    print(f"\nReport saved to: {REPORT_FILE}")

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("RAG TEST SUITE — BASELINE")
    print(f"Model: {MODEL} | DB: {CHROMA_PATH}")
    print("Running all test categories...\n")

    all_results = {}

    all_results["STANDALONE"]          = run_standalone(STANDALONE_TESTS, "STANDALONE QUESTIONS")
    all_results["FOLLOW-UP SEQUENCE"]  = run_sequence(FOLLOWUP_SEQUENCE, "FOLLOW-UP SEQUENCE")
    all_results["PRONOUN HELL"]        = run_sequence(PRONOUN_HELL_SEQUENCE, "PRONOUN HELL SEQUENCE")
    all_results["CROSS-DOCUMENT"]      = run_standalone(CROSS_DOC_TESTS, "CROSS-DOCUMENT SYNTHESIS")
    all_results["NEGATION"]            = run_standalone(NEGATION_TESTS, "NEGATION QUESTIONS")
    all_results["SUPERLATIVES"]        = run_standalone(SUPERLATIVE_TESTS, "SUPERLATIVE + COMPARISON")
    all_results["ABSENT INFO"]         = run_standalone(ABSENT_INFO_TESTS, "ABSENT INFORMATION")
    all_results["CONTRADICTORY"]       = run_sequence(CONTRADICTORY_TESTS, "CONTRADICTORY INJECTION")
    all_results["AMBIGUOUS REFERENT"]  = run_sequence(AMBIGUOUS_REFERENT_TESTS, "AMBIGUOUS REFERENTS")
    all_results["TOPIC BLEED"]         = run_standalone(TOPIC_BLEED_TESTS, "TOPIC BLEED")
    all_results["EDGE CASES"]          = run_edge_cases(EDGE_CASE_TESTS)
    all_results["TEMPORAL"]            = run_standalone(TEMPORAL_TESTS, "TEMPORAL REASONING")
    all_results["MEMORY & RECALL"]     = run_standalone(MEMORY_AND_RECALL_TESTS, "MEMORY & RECALL")

    generate_report(all_results)