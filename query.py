import chromadb
import requests
import json
import uuid
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

embedder  = SentenceTransformer('all-MiniLM-L6-v2')
reranker  = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
client    = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("knowledge")
RELEVANCE_THRESHOLD = -5.0  # CrossEncoder scores below this = likely irrelevant


# ── TokenLedger ───────────────────────────────────────────────────────────────

class TokenLedger:
    COMPONENTS = ["rewriter", "generation", "memory_inject", "distillation"]

    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.started_at = datetime.now().isoformat()
        self.exchanges  = 0
        self.components = {c: {"calls": 0, "tokens": 0} for c in self.COMPONENTS}

    def charge(self, component, tokens):
        """Charge tokens to a component."""
        if component not in self.components:
            self.components[component] = {"calls": 0, "tokens": 0}
        self.components[component]["calls"]  += 1
        self.components[component]["tokens"] += tokens

    @property
    def total(self):
        return sum(v["tokens"] for v in self.components.values())

    def exchange_summary(self):
        """Print per-exchange token breakdown."""
        parts = []
        for c, v in self.components.items():
            if v["tokens"] > 0:
                parts.append(f"{c}: {v['tokens']}")
        print(f"[Tokens] {' | '.join(parts)} | session_total: {self.total}")

    def session_summary(self):
        """Print full session summary on exit."""
        print("\n" + "=" * 50)
        print(f"SESSION TOKEN SUMMARY  (id: {self.session_id})")
        print(f"Started : {self.started_at}")
        print(f"Ended   : {datetime.now().isoformat()}")
        print(f"Exchanges: {self.exchanges}")
        print("-" * 50)
        for c, v in self.components.items():
            if v["calls"] > 0:
                avg = round(v["tokens"] / v["calls"])
                print(f"  {c:<16} calls:{v['calls']:>3}  tokens:{v['tokens']:>6}  avg/call:{avg:>4}")
        print(f"  {'TOTAL':<16}              tokens:{self.total:>6}")
        print("=" * 50)

    def to_dict(self):
        return {
            "session_id":  self.session_id,
            "started_at":  self.started_at,
            "ended_at":    datetime.now().isoformat(),
            "exchanges":   self.exchanges,
            "components":  self.components,
            "total_tokens": self.total
        }

ledger = TokenLedger()

# ── Layer 1: Working Memory ───────────────────────────────────────────────────

RECENT_MEMORY_PATH = "./memory/recent.json"

def load_recent_memory():
    """Load last 2 exchanges from disk on startup."""
    if not os.path.exists(RECENT_MEMORY_PATH):
        return []
    try:
        with open(RECENT_MEMORY_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[Memory] Loaded {len(data)} exchange(s) from previous session.")
        return data
    except Exception as e:
        print(f"[Memory] Could not load recent memory: {e}")
        return []

def save_recent_memory(history):
    """Save last 2 exchanges to disk after every exchange."""
    os.makedirs("./memory", exist_ok=True)
    try:
        with open(RECENT_MEMORY_PATH, 'w', encoding='utf-8') as f:
            json.dump(history[-2:], f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Memory] Could not save recent memory: {e}")

conversation_history = load_recent_memory()

# ── Layer 2: Episodic Memory ──────────────────────────────────────────────────

FULLLOG_PATH    = "./memory/fulllog.jsonl"
FULLLOG_CAPACITY = 100

def load_fulllog():
    """Load all exchanges from fulllog."""
    if not os.path.exists(FULLLOG_PATH):
        return []
    try:
        with open(FULLLOG_PATH, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"[Memory] Could not load fulllog: {e}")
        return []

def save_fulllog(exchanges):
    """Write all exchanges back to fulllog."""
    os.makedirs("./memory", exist_ok=True)
    try:
        with open(FULLLOG_PATH, 'w', encoding='utf-8') as f:
            for ex in exchanges:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"[Memory] Could not save fulllog: {e}")

def append_to_fulllog(exchange_data):
    """Append one exchange, drop oldest if over capacity."""
    exchanges = load_fulllog()
    exchanges.append(exchange_data)
    if len(exchanges) > FULLLOG_CAPACITY:
        dropped = len(exchanges) - FULLLOG_CAPACITY
        exchanges = exchanges[dropped:]  # drop oldest
    save_fulllog(exchanges)


# ── Layer 2.5: Queryable Conversation History ─────────────────────────────────

conv_collection = client.get_or_create_collection("conversations")

HISTORY_SIGNALS = [
    'did i', 'what did i', 'did we', 'have i', 'have we',
    'earlier', 'before', 'last time', 'previously',
    'we discussed', 'we talked about', 'you told me',
    'i asked', 'i mentioned', 'my previous', 'in the past', 'from before'
]

COMPARISON_SIGNALS = [
    'what i said', 'what i concluded', 'what i mentioned',
    'my previous answer', 'my view', 'my conclusion',
    'does this match what i', 'consistent with what i said',
    'contradict what i', 'still hold up', 'does that still hold'
]

def is_history_query(question):
    q = question.lower()
    return any(signal in q for signal in HISTORY_SIGNALS)

def is_comparison_query(question):
    q = question.lower()
    return any(signal in q for signal in COMPARISON_SIGNALS)


def embed_exchange_to_conversations(exchange_record):
    """Embed one exchange into the permanent, unbounded conversations collection."""
    text_to_embed = f"Q: {exchange_record['question']}\nA: {exchange_record['answer']}"
    vec = embedder.encode(text_to_embed).tolist()
    conv_id = f"{exchange_record['session_id']}_{exchange_record['exchange_num']}_{exchange_record['timestamp']}"
    try:
        conv_collection.add(
            documents=[text_to_embed],
            embeddings=[vec],
            ids=[conv_id],
            metadatas=[{
                "timestamp": exchange_record["timestamp"],
                "session_id": exchange_record["session_id"],
                "sources": ",".join(exchange_record.get("sources", [])),
            }]
        )
    except Exception as e:
        print(f"[Memory] Could not embed exchange into conversations collection: {e}")


def search_conversations(question, n=4):
    """Return raw docs from the conversations collection, no generation."""
    vec = embedder.encode(question).tolist()
    try:
        results = conv_collection.query(
            query_embeddings=[vec], n_results=n,
            include=["documents", "metadatas"]
        )
        return results['documents'][0], results['metadatas'][0]
    except Exception as e:
        print(f"[Memory] Could not search conversations: {e}")
        return [], []


def search_knowledge(question, n=4):
    """Return raw docs from the knowledge collection, no generation."""
    vec = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[vec], n_results=n,
        include=["documents", "metadatas"]
    )
    return results['documents'][0], results['ids'][0]


def query_conversations_or_both(question, ex_ledger, dual=False):
    """Search conversations, or both conversations + knowledge if dual=True."""
    conv_docs, conv_metas = search_conversations(question)

    context_parts = []
    if conv_docs:
        context_parts.append("From your past conversations:\n" + "\n\n".join(conv_docs))
    else:
        context_parts.append("From your past conversations: (no matches found)")

    if dual:
        kb_docs, kb_ids = search_knowledge(question)
        if kb_docs:
            context_parts.append("From the knowledge base articles:\n" + "\n\n".join(kb_docs[:2]))

    context = "\n\n---\n\n".join(context_parts)

    if not conv_docs and not dual:
        print("\nAnswer: No matching past conversation found.")
        return

    prompt = f"""Answer the question using the labeled sources below. Keep the sources distinct in your answer — 
don't blend a past conversation's claim with an article's claim as if they're the same thing.
If nothing here answers the question, say so plainly.

{context}

Question: {question}"""

    response = requests.post('http://localhost:11434/api/generate',
        json={"model": "llama3.2", "prompt": prompt, "stream": True},
        stream=True
    )

    print("\nAnswer: ", end="", flush=True)
    full_answer = ""
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            token = chunk.get('response', '')
            print(token, end="", flush=True)
            full_answer += token
            if chunk.get('done'):
                ex_ledger.charge("generation",
                    chunk.get('prompt_eval_count', 0) + chunk.get('eval_count', 0))
                break
    print()

    if conv_metas:
        print(f"\nFrom {len(conv_docs)} past exchange(s):")
        for meta in conv_metas:
            print(f"  [{meta['timestamp'][:10]}] session {meta['session_id']}")

# ── Layer 3: Semantic Memory (personality profile) ────────────────────────────

PROFILE_PATH    = "./memory/profile.json"
DISTILL_EVERY   = 25
PROFILE_TOKEN_CAP = 700

DEFAULT_PROFILE = {
    "topics": {},
    "thinking_style": "",
    "recurring_conclusions": [],
    "last_updated": None,
    "exchanges_since_update": 0,
    "total_distillations": 0
}

def load_profile():
    if not os.path.exists(PROFILE_PATH):
        return dict(DEFAULT_PROFILE)
    try:
        with open(PROFILE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[Memory] Could not load profile: {e}")
        return dict(DEFAULT_PROFILE)

def save_profile(profile):
    os.makedirs("./memory", exist_ok=True)
    try:
        with open(PROFILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Memory] Could not save profile: {e}")

def compress_exchange(ex):
    """Shrink a fulllog entry down to the essentials for distillation input."""
    return {
        "q": ex.get("question", "")[:120],
        "a": ex.get("answer", "")[:120],
        "sources": ex.get("sources", [])
    }

def distill_profile(ex_ledger=None):
    """Run distillation: read fulllog + existing profile, update profile."""
    exchanges = load_fulllog()
    if not exchanges:
        return

    profile = load_profile()
    compressed = [compress_exchange(e) for e in exchanges]
    exchanges_text = "\n".join(
        f"Q: {c['q']} | A: {c['a']} | sources: {c['sources']}" for c in compressed
    )

    prompt = f"""You are updating a personality/preference profile based on conversation history.

EXISTING PROFILE:
{json.dumps(profile, ensure_ascii=False)}

NEW EXCHANGES (most recent {len(compressed)}):
{exchanges_text}

Update the profile by analyzing:
1. Topic depth preference — does the user want deep technical detail or high-level summaries, per topic?
2. Preferred format per topic — comparative tables, narrative explanation, data-first, conceptual?
3. Recurring conclusions — positions or opinions the user has reached and restated
4. Reasoning style — do they ask why before how, do they challenge assumptions, do they want brevity or depth?

RULE: If new exchanges contradict the existing profile, the new pattern wins. Update, don't just append.
Keep the entire profile under {PROFILE_TOKEN_CAP} tokens. Be specific, not generic — avoid vague filler like "curious and analytical."

Return ONLY valid JSON matching this exact structure, nothing else:
{{
  "topics": {{"topic_name": {{"depth": "...", "format": "...", "notes": "..."}}}},
  "thinking_style": "...",
  "recurring_conclusions": ["...", "..."],
  "last_updated": null,
  "exchanges_since_update": 0,
  "total_distillations": 0
}}"""

    response = requests.post('http://localhost:11434/api/generate',
        json={"model": "llama3.2", "prompt": prompt, "stream": False}
    )
    data = response.json()
    raw_output = data.get('response', '').strip()
    tokens = data.get('prompt_eval_count', 0) + data.get('eval_count', 0)

    if ex_ledger:
        ex_ledger.charge("distillation", tokens)
    else:
        ledger.charge("distillation", tokens)

    # Try to parse JSON out of the model output (strip markdown fences if present)
    cleaned = raw_output.replace('```json', '').replace('```', '').strip()
    try:
        new_profile = json.loads(cleaned)
        new_profile["last_updated"] = datetime.now().isoformat()
        new_profile["exchanges_since_update"] = 0
        new_profile["total_distillations"] = profile.get("total_distillations", 0) + 1
        save_profile(new_profile)
        print(f"[Memory] Profile distilled and updated. (distillation #{new_profile['total_distillations']}, tokens: {tokens})")
    except json.JSONDecodeError:
        print(f"[Memory] Distillation output was not valid JSON, profile not updated. Tokens spent: {tokens}")

def maybe_distill(ex_ledger):
    profile = load_profile()
    profile["exchanges_since_update"] = profile.get("exchanges_since_update", 0) + 1
    save_profile(profile)
    if profile["exchanges_since_update"] >= DISTILL_EVERY:
        print(f"[Memory] {DISTILL_EVERY} exchanges reached — running distillation...")
        distill_profile(ex_ledger)

def get_relevant_profile_snippet(question, profile):
    """Return only the profile section relevant to the current question's topic, if any match."""
    if not profile.get("topics"):
        return ""
    q_lower = question.lower()
    matched = {}
    for topic, info in profile["topics"].items():
        if topic.lower() in q_lower:
            matched[topic] = info
    if not matched:
        return ""
    snippet = "User preference context:\n"
    for topic, info in matched.items():
        snippet += f"- {topic}: depth={info.get('depth','')}, format={info.get('format','')}, notes={info.get('notes','')}\n"
    if profile.get("thinking_style"):
        snippet += f"Thinking style: {profile['thinking_style']}\n"
    return snippet

# ── per-exchange ledger snapshot (resets each turn) ───────────────────────────

class ExchangeLedger:
    """Tracks tokens for a single exchange, then merges into session ledger."""
    def __init__(self):
        self.data = {c: 0 for c in TokenLedger.COMPONENTS}

    def charge(self, component, tokens):
        self.data[component] = self.data.get(component, 0) + tokens

    def commit(self, session_ledger):
        for component, tokens in self.data.items():
            if tokens > 0:
                session_ledger.charge(component, tokens)

    def summary(self):
        parts = [f"{c}: {t}" for c, t in self.data.items() if t > 0]
        total = sum(self.data.values())
        return f"[Tokens] {' | '.join(parts)} | exchange_total: {total}"

# ── core functions ────────────────────────────────────────────────────────────

def needs_rewrite(question, history, ex_ledger):
    """Check if question references prior context via pronouns or implicit comparison words."""
    if not history:
        return False

    trigger_words = [
        'it', 'its', 'they', 'their', 'them', 'that', 'this',
        'he', 'she', 'his', 'her', 'those', 'these', 'there',
        'differently', 'instead', 'similarly', 'also', 'too',
        'compared', 'comparison', 'versus', 'vs', 'other', 'others'
    ]
    words = question.lower().replace('?', '').split()
    return any(w in trigger_words for w in words)

def rewrite_question(question, history, ex_ledger):
    if not history:
        return question, False

    if not needs_rewrite(question, history, ex_ledger):
        return question, False

    recent = history[-2:]
    history_text = ""
    for q, a in recent:
        history_text += f"Q: {q}\nA: {a[:150]}\n\n"

    prompt = f"""Given this conversation history:
    {history_text}
    Rewrite this follow-up question as a fully self-contained question.
    Rules:
    - If the question already names its subject, KEEP that exact subject. Only ADD the missing context needed to understand it standalone.
    - If the question uses a pronoun or has no named subject, REPLACE the pronoun with the correct entity from history.
    - Do NOT answer the question or embed the answer into the rewrite.
    - Keep it as an open-ended question starting with What, How, Why, Who, or Which — never Yes/No.
    - Preserve the original intent exactly.

    Question to rewrite: "{question}"
    Return ONLY the rewritten question, nothing else."""

    response = requests.post('http://localhost:11434/api/generate',
        json={"model": "llama3.2", "prompt": prompt, "stream": False}
    )
    data = response.json()
    rewritten = data.get('response', question).strip()
    tokens = data.get('prompt_eval_count', 0) + data.get('eval_count', 0)
    ex_ledger.charge("rewriter", tokens)
    return rewritten, rewritten != question


def query_knowledge(question, history):
    # Input validation
    has_real_words = any(c.isalpha() for c in question)
    if not has_real_words and not history:
        print("(Please ask a question.)")
        return

    ex_ledger = ExchangeLedger()

    # Rewrite
    rewritten, was_rewritten = rewrite_question(question, history, ex_ledger)
    if was_rewritten:
        print(f"[Rewritten]: {rewritten}")

    # Retrieve
    question_embedding = embedder.encode(rewritten).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=6,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = results['documents'][0]
    pairs  = [[rewritten, chunk] for chunk in retrieved_chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, retrieved_chunks), reverse=True)
    top_score = ranked[0][0]
    print(f"[DEBUG] top rerank score: {top_score}")
    retrieved_chunks = [chunk for _, chunk in ranked[:2]]
    ids    = results['ids'][0]
    context = "\n\n".join(retrieved_chunks)
    sources = list(set([id.rsplit('_', 1)[0] for id in ids]))

    # Relevance gate
    if top_score < RELEVANCE_THRESHOLD:
        print(f"\nAnswer: I don't have relevant information in the knowledge base to answer that. (top relevance score: {top_score:.2f})")
        history.append((question, "I don't have relevant information in the knowledge base to answer that."))
        save_recent_memory(history)
        ex_ledger.commit(ledger)
        ledger.exchanges += 1
        return

    # Layer 3: inject relevant profile context if available
    profile = load_profile()
    profile_snippet = get_relevant_profile_snippet(rewritten, profile)
    profile_tokens_estimate = len(profile_snippet.split()) * 1.3  # rough estimate before call

    context_block = context
    if profile_snippet:
        context_block = f"{profile_snippet}\n\n{context}"
        ex_ledger.charge("memory_inject", int(profile_tokens_estimate))

    # Generate
    response = requests.post('http://localhost:11434/api/generate',
        json={
            "model": "llama3.2",
            "prompt":f"""Answer based only on this context. If the context does not contain enough information to answer the question, respond with exactly: "I don't have relevant information in the knowledge base to answer that." Do not guess, speculate, or use outside knowledge.
                Context:
                {context_block}

                Question: {rewritten}""",   
            "stream": True
        },
        stream=True
    )

    print("\nAnswer: ", end="", flush=True)
    full_answer = ""
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            token = chunk.get('response', '')
            print(token, end="", flush=True)
            full_answer += token
            if chunk.get('done'):
                ex_ledger.charge("generation",
                    chunk.get('prompt_eval_count', 0) + chunk.get('eval_count', 0))
                break
    print()

    # Commit to session ledger + print
    ex_ledger.commit(ledger)
    ledger.exchanges += 1
    print(ex_ledger.summary())

    # Update history
    history.append((question, full_answer))
    save_recent_memory(history)

    # Build exchange record and append to fulllog
    exchange_record = {
        "timestamp":    datetime.now().isoformat(),
        "session_id":   ledger.session_id,
        "exchange_num": ledger.exchanges,
        "question":     question,
        "rewritten":    rewritten,
        "answer":       full_answer,
        "sources":      sources,
        "tokens": {
            "rewriter":   ex_ledger.data.get("rewriter", 0),
            "generation": ex_ledger.data.get("generation", 0),
            "total":      sum(ex_ledger.data.values())
        }
    }
    append_to_fulllog(exchange_record)
    embed_exchange_to_conversations(exchange_record)
    maybe_distill(ex_ledger)

    print(f"\nSources: {', '.join(sources)}")
    print("\nRelevant excerpts:")
    for chunk, source in zip(retrieved_chunks, ids):
        source_name = source.rsplit('_', 1)[0]
        print(f"\n[{source_name}]: ...{chunk[:200]}...")


# ── main loop ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Knowledge Base Ready. Type 'exit' to quit.\n")
    try:
        while True:
            try:
                question = input("You: ")
            except KeyboardInterrupt:
                print("\n(Use 'exit' to quit cleanly)")
                break
            if question.lower() == 'exit':
                break

            # Explicit override
            if question.lower().startswith('recall:'):
                actual_q = question[len('recall:'):].strip()
                ex_ledger = ExchangeLedger()
                query_conversations_or_both(actual_q, ex_ledger, dual=is_comparison_query(actual_q))
                ex_ledger.commit(ledger)
                ledger.exchanges += 1
                print(ex_ledger.summary())

            # Deterministic auto-detect
            elif is_history_query(question) or is_comparison_query(question):
                ex_ledger = ExchangeLedger()
                query_conversations_or_both(question, ex_ledger, dual=is_comparison_query(question))
                ex_ledger.commit(ledger)
                ledger.exchanges += 1
                print(ex_ledger.summary())

            # Normal knowledge-base flow
            else:
                query_knowledge(question, conversation_history)
    finally:
        ledger.session_summary()