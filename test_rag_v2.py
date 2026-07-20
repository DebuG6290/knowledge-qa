import io
import re
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime

# Import the REAL functions from query.py — not a reimplementation
import query as q

REPORT_FILE = f"test_report_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


# ── capture helper ────────────────────────────────────────────────────────────

def run_captured(fn, *args, **kwargs):
    """Run a query.py function, capture everything it prints, return (captured_text, elapsed)."""
    buf = io.StringIO()
    start = time.time()
    with redirect_stdout(buf):
        fn(*args, **kwargs)
    elapsed = round(time.time() - start, 1)
    return buf.getvalue(), elapsed


def parse_answer(captured):
    """Pull the Answer: ... block out of captured stdout."""
    m = re.search(r"Answer:\s*(.*?)(?:\n\[Tokens\]|\nSources:|\nFrom \d|\Z)", captured, re.S)
    return m.group(1).strip() if m else ""


def parse_sources(captured):
    m = re.search(r"Sources:\s*(.*)", captured)
    if not m:
        return []
    return [s.strip() for s in m.group(1).split(",") if s.strip()]


def parse_rewritten(captured):
    m = re.search(r"\[Rewritten\]:\s*(.*)", captured)
    return m.group(1).strip() if m else None


def parse_tokens(captured):
    """Sum every [Tokens] ... exchange_total / session_total line in this capture."""
    matches = re.findall(r"exchange_total:\s*(\d+)", captured)
    if matches:
        return sum(int(m) for m in matches)
    matches = re.findall(r"generation:\s*(\d+)", captured)
    return sum(int(m) for m in matches)


def parse_relevance_score(captured):
    m = re.search(r"top rerank score:\s*(-?\d+\.?\d*)", captured)
    return float(m.group(1)) if m else None


def parse_route(captured):
    """Infer which path was taken based on captured output shape."""
    if "From your past conversations" in captured or "past exchange(s)" in captured:
        return "conversations"
    if "No matching past conversation found" in captured:
        return "conversations"
    return "knowledge"


def check_keywords(text, keywords):
    text_lower = text.lower()
    return {kw: kw.lower() in text_lower for kw in keywords}


def uncertainty_expressed(answer):
    signals = [
        "don't know", "do not know", "not in", "cannot find",
        "no information", "not mentioned", "not provided",
        "i'm not sure", "i am not sure", "unclear", "not available",
        "context does not", "context doesn't", "not enough",
        "no relevant information", "don't have relevant"
    ]
    a = answer.lower()
    return any(s in a for s in signals)


# ── test definitions (same 31 as before, kept identical for a clean comparison) ─

STANDALONE_TESTS = [
    {"id": "S1", "question": "What is Bridgewater's AI strategy?",
     "expected_sources": ["asset_management"], "expected_keywords": ["PrinciplesOS", "Ray Dalio", "trade"]},
    {"id": "S2", "question": "Who founded Bridgewater?",
     "expected_sources": ["asset_management"], "expected_keywords": ["Ray Dalio", "1975"]},
    {"id": "S3", "question": "What does Vanguard use AI for?",
     "expected_sources": ["asset_management"], "expected_keywords": ["client", "virtual assistant"]},
    {"id": "S4", "question": "What is BlackRock's Aladdin system?",
     "expected_sources": ["asset_management"], "expected_keywords": ["risk", "BlackRock", "Aladdin"]},
    {"id": "S5", "question": "How does JPMorgan use AI?",
     "expected_sources": ["article"], "expected_keywords": ["JPMorgan"]},
]

FOLLOWUP_SEQUENCE = [
    {"id": "F1", "question": "What is Bridgewater's AI strategy?",
     "expected_keywords": ["PrinciplesOS", "Ray Dalio"], "rewrite_should_contain": []},
    {"id": "F2", "question": "How is that different from BlackRock?",
     "expected_keywords": ["BlackRock", "Bridgewater"], "rewrite_should_contain": ["Bridgewater", "BlackRock"]},
    {"id": "F3", "question": "What about Vanguard?",
     "expected_keywords": ["Vanguard"], "rewrite_should_contain": []},
    {"id": "F4", "question": "Which of these three is most sustainable long term and why?",
     "expected_keywords": ["sustainable"], "rewrite_should_contain": []},
    {"id": "F5", "question": "Going back to Bridgewater, who founded it?",
     "expected_keywords": ["Ray Dalio", "1975"], "rewrite_should_contain": ["Bridgewater"]},
    {"id": "F6", "question": "Why did experts say that approach gets arbitraged away?",
     "expected_keywords": ["arbitraged"], "rewrite_should_contain": []},
]

PRONOUN_HELL_SEQUENCE = [
    {"id": "P1", "question": "What is Bridgewater's investment strategy?", "expected_keywords": ["Bridgewater"]},
    {"id": "P2", "question": "How did they develop it?", "expected_keywords": ["Bridgewater"]},
    {"id": "P3", "question": "Who was behind that decision?", "expected_keywords": ["Ray Dalio"]},
    {"id": "P4", "question": "Did it work?", "expected_keywords": ["Bridgewater"]},
    {"id": "P5", "question": "What did critics say about it?", "expected_keywords": ["Bridgewater"]},
    {"id": "P6", "question": "Why did they persist anyway?", "expected_keywords": ["Bridgewater"]},
]

CROSS_DOC_TESTS = [
    {"id": "C1", "question": "How does the AI technique Bridgewater uses relate to how neural networks actually work?",
     "expected_keywords": ["Bridgewater", "neural"]},
    {"id": "C2", "question": "What Marcus Aurelius principle best applies to BlackRock's risk management approach?",
     "expected_keywords": ["BlackRock", "risk"]},
]

NEGATION_TESTS = [
    {"id": "N1", "question": "What does Bridgewater NOT use AI for?", "expected_keywords": ["Bridgewater"]},
    {"id": "N2", "question": "Which firm does not focus on alpha generation?", "expected_keywords": ["Vanguard"]},
]

SUPERLATIVE_TESTS = [
    {"id": "SU1", "question": "Which firm manages the most assets?", "expected_keywords": ["BlackRock", "trillion"]},
    {"id": "SU2", "question": "What is the biggest risk across all three firms?", "expected_keywords": ["risk"]},
]

ABSENT_INFO_TESTS = [
    {"id": "A1", "question": "What is BlackRock's revenue in 2023?", "should_answer": False},
    {"id": "A2", "question": "How many engineers work on PrinciplesOS?", "should_answer": False},
    {"id": "A3", "question": "What is the capital of France?", "should_answer": False},
    {"id": "A4", "question": "Who is Virat Kohli?", "should_answer": False},
]

CONTRADICTORY_TESTS = [
    {"id": "CO1", "question": "Who founded Bridgewater?", "expected_keywords": ["Ray Dalio"]},
    {"id": "CO2", "question": "Actually I read it was founded by John Smith in 1980", "expected_keywords": ["Ray Dalio"]},
    {"id": "CO3", "question": "So who actually founded Bridgewater?", "expected_keywords": ["Ray Dalio"]},
]

AMBIGUOUS_REFERENT_TESTS = [
    {"id": "AM1", "question": "Compare BlackRock and Bridgewater's AI approaches", "expected_keywords": ["BlackRock", "Bridgewater"]},
    {"id": "AM2", "question": "Which one uses more data?", "expected_keywords": []},
    {"id": "AM3", "question": "Does it have more employees?", "expected_keywords": []},
]

TOPIC_BLEED_TESTS = [
    {"id": "TB1", "question": "What principles guide Bridgewater's decision making?",
     "forbidden_sources": ["marcus_aurelius"], "expected_keywords": ["Ray Dalio", "PrinciplesOS"]},
    {"id": "TB2", "question": "How should one approach adversity according to the texts?", "expected_keywords": []},
]

EDGE_CASE_TESTS = [
    {"id": "E1", "question": "", "note": "Empty input"},
    {"id": "E2", "question": "?", "note": "Just punctuation"},
    {"id": "E3", "question": "asdfjkl;", "note": "Gibberish"},
    {"id": "E4", "question": "What?", "note": "Single word follow-up"},
    {"id": "E5", "question": "   ", "note": "Whitespace only"},
]

TEMPORAL_TESTS = [
    {"id": "T1", "question": "Which AI initiative came first, BlackRock's Aladdin or Bridgewater's PrinciplesOS?",
     "expected_keywords": ["BlackRock", "Bridgewater"]},
]

# ── new: memory & recall tests, calling the REAL routing functions ────────────

MEMORY_TESTS = [
    {"id": "M1", "question": "recall: what did I conclude about Bridgewater?",
     "expect_route": "conversations"},
    {"id": "M2", "question": "What did I say about Vanguard earlier?",
     "expect_route": "conversations"},
    {"id": "M3", "question": "Does what I said about Vanguard match the article?",
     "expect_route": "conversations"},  # dual search still flows through query_conversations_or_both
    {"id": "M4", "question": "What is the capital of France?",
     "expect_zero_generation": True},
    {"id": "M5", "question": "Does BlackRock's approach match Bridgewater's?",
     "note": "Known ceiling (Part 14): may over-refuse despite sufficient retrieved context"},
]


# ── runners ───────────────────────────────────────────────────────────────────

def run_standalone(tests, label):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    results = []
    for t in tests:
        print(f"\n[{t['id']}] {t['question'][:80]}")
        history = []  # fresh, no prior context — true standalone
        captured, elapsed = run_captured(q.query_knowledge, t['question'], history)
        answer  = parse_answer(captured)
        sources = parse_sources(captured)
        tokens  = parse_tokens(captured)

        kw_results = check_keywords(answer, t.get('expected_keywords', []))
        kw_pass = all(kw_results.values()) if kw_results else None
        src_expected = t.get('expected_sources', [])
        src_hit = any(s in sources for s in src_expected) if src_expected else None
        forbidden = t.get('forbidden_sources', [])
        src_clean = not any(s in sources for s in forbidden) if forbidden else True
        should_answer = t.get('should_answer', True)
        uncertainty = uncertainty_expressed(answer)

        if should_answer:
            overall = "PASS" if (kw_pass is not False and src_hit is not False and src_clean) else "FAIL"
        else:
            overall = "PASS" if uncertainty else "FAIL"

        print(f"   Answer: {answer[:150]}")
        print(f"   Sources: {sources} | Tokens: {tokens} | {elapsed}s | {overall}")
        results.append({"id": t['id'], "question": t['question'], "answer": answer,
                         "sources": sources, "tokens": tokens, "overall": overall})
    return results


def run_sequence(tests, label):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    results = []
    history = []
    for t in tests:
        print(f"\n[{t['id']}] {t['question'][:80]}")
        captured, elapsed = run_captured(q.query_knowledge, t['question'], history)
        answer    = parse_answer(captured)
        sources   = parse_sources(captured)
        rewritten = parse_rewritten(captured)
        tokens    = parse_tokens(captured)

        kw_results = check_keywords(answer, t.get('expected_keywords', []))
        kw_pass = all(kw_results.values()) if kw_results else None
        rw_check = t.get('rewrite_should_contain', [])
        rw_pass = all(kw.lower() in (rewritten or "").lower() for kw in rw_check) if rw_check else None

        overall = "PASS" if (kw_pass is not False and rw_pass is not False) else "FAIL"

        print(f"   Rewritten: {rewritten}")
        print(f"   Answer: {answer[:150]}")
        print(f"   Tokens: {tokens} | {elapsed}s | {overall}")
        results.append({"id": t['id'], "question": t['question'], "rewritten": rewritten,
                         "answer": answer, "tokens": tokens, "overall": overall})
    return results


def run_edge_cases(tests):
    print(f"\n{'='*60}\n  EDGE CASES\n{'='*60}")
    results = []
    for t in tests:
        print(f"\n[{t['id']}] '{t['question']}' — {t['note']}")
        history = []
        try:
            captured, elapsed = run_captured(q.query_knowledge, t['question'], history)
            if "(Please ask a question.)" in captured:
                print("   HANDLED before retrieval")
                results.append({"id": t['id'], "overall": "HANDLED"})
                continue
            answer = parse_answer(captured)
            uncertainty = uncertainty_expressed(answer)
            overall = "HANDLED" if uncertainty else "FAIL — answered confidently"
            print(f"   Answer: {answer[:150]} | {overall}")
            results.append({"id": t['id'], "overall": overall, "answer": answer})
        except Exception as e:
            print(f"   CRASHED: {e}")
            results.append({"id": t['id'], "overall": f"CRASH: {e}"})
    return results


def run_memory_tests(tests):
    """Calls the REAL routing logic: recall: prefix, auto-detect, dual search, relevance gate."""
    print(f"\n{'='*60}\n  MEMORY & RECALL (real routing)\n{'='*60}")
    results = []
    for t in tests:
        print(f"\n[{t['id']}] {t['question']}")
        if t.get('note'):
            print(f"   NOTE: {t['note']}")

        question = t['question']
        ex_ledger = q.ExchangeLedger()

        if question.lower().startswith('recall:'):
            actual_q = question[len('recall:'):].strip()
            captured, elapsed = run_captured(
                q.query_conversations_or_both, actual_q, ex_ledger,
                dual=q.is_comparison_query(actual_q)
            )
            route = parse_route(captured)
        elif q.is_history_query(question) or q.is_comparison_query(question):
            captured, elapsed = run_captured(
                q.query_conversations_or_both, question, ex_ledger,
                dual=q.is_comparison_query(question)
            )
            route = parse_route(captured)
        else:
            history = []
            captured, elapsed = run_captured(q.query_knowledge, question, history)
            route = "knowledge"

        answer = parse_answer(captured)
        tokens = parse_tokens(captured)
        rel_score = parse_relevance_score(captured)

        route_pass = None
        if 'expect_route' in t:
            route_pass = (route == t['expect_route'])

        zero_gen_pass = None
        if t.get('expect_zero_generation'):
            zero_gen_pass = (tokens == 0)

        checks = [c for c in [route_pass, zero_gen_pass] if c is not None]
        overall = "PASS" if (checks and all(checks)) else ("PASS" if not checks else "FAIL")

        print(f"   Route detected: {route} | Tokens: {tokens} | Relevance score: {rel_score}")
        print(f"   Answer: {answer[:150]}")
        print(f"   {overall}")

        results.append({"id": t['id'], "question": question, "answer": answer,
                         "route": route, "tokens": tokens, "relevance_score": rel_score,
                         "overall": overall, "note": t.get('note', '')})
    return results


# ── report ────────────────────────────────────────────────────────────────────

def generate_report(all_results):
    lines = []
    lines.append("=" * 70)
    lines.append("RAG SYSTEM TEST REPORT — POST-MEMORY (real query.py functions)")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    total_pass, total_fail, total_tokens = 0, 0, 0

    for section, results in all_results.items():
        if not results:
            continue
        lines.append(f"\n── {section} ──")
        s_pass = sum(1 for r in results if r.get('overall') in ['PASS', 'HANDLED'])
        s_fail = sum(1 for r in results if r.get('overall') not in ['PASS', 'HANDLED'])
        lines.append(f"Score: {s_pass}/{len(results)}")
        for r in results:
            lines.append(f"  [{r['id']}] {r.get('overall','?')}")
            if r.get('note'):
                lines.append(f"        Note: {r['note']}")
            if r.get('answer'):
                lines.append(f"        Answer: {r['answer'][:100]}")
            if r.get('rewritten'):
                lines.append(f"        Rewritten: {r['rewritten'][:100]}")
            if r.get('route'):
                lines.append(f"        Route: {r['route']}")
            tok = r.get('tokens', 0)
            if tok:
                total_tokens += tok
                lines.append(f"        Tokens: {tok}")
        total_pass += s_pass
        total_fail += s_fail

    lines.append("\n" + "=" * 70)
    lines.append("SUMMARY")
    lines.append(f"Total PASS: {total_pass}")
    lines.append(f"Total FAIL: {total_fail}")
    lines.append(f"Overall: {total_pass}/{total_pass+total_fail}")
    lines.append(f"Total tokens consumed this run: {total_tokens}")
    lines.append("=" * 70)

    report = "\n".join(lines)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print("\n\n" + report)
    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    print("RAG TEST SUITE v2 — calling REAL query.py functions, not a reimplementation")
    all_results = {}
    all_results["STANDALONE"]         = run_standalone(STANDALONE_TESTS, "STANDALONE QUESTIONS")
    all_results["FOLLOW-UP SEQUENCE"] = run_sequence(FOLLOWUP_SEQUENCE, "FOLLOW-UP SEQUENCE")
    all_results["PRONOUN HELL"]       = run_sequence(PRONOUN_HELL_SEQUENCE, "PRONOUN HELL")
    all_results["CROSS-DOCUMENT"]     = run_standalone(CROSS_DOC_TESTS, "CROSS-DOCUMENT SYNTHESIS")
    all_results["NEGATION"]           = run_standalone(NEGATION_TESTS, "NEGATION")
    all_results["SUPERLATIVES"]       = run_standalone(SUPERLATIVE_TESTS, "SUPERLATIVES")
    all_results["ABSENT INFO"]        = run_standalone(ABSENT_INFO_TESTS, "ABSENT INFORMATION")
    all_results["CONTRADICTORY"]      = run_sequence(CONTRADICTORY_TESTS, "CONTRADICTORY INJECTION")
    all_results["AMBIGUOUS REFERENT"] = run_sequence(AMBIGUOUS_REFERENT_TESTS, "AMBIGUOUS REFERENTS")
    all_results["TOPIC BLEED"]        = run_standalone(TOPIC_BLEED_TESTS, "TOPIC BLEED")
    all_results["EDGE CASES"]         = run_edge_cases(EDGE_CASE_TESTS)
    all_results["TEMPORAL"]           = run_standalone(TEMPORAL_TESTS, "TEMPORAL")
    all_results["MEMORY & RECALL"]    = run_memory_tests(MEMORY_TESTS)

    generate_report(all_results)