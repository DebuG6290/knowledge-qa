"""
One-time backfill — embeds existing fulllog.jsonl entries into the
conversations ChromaDB collection, since they predate the recall feature.
Run this once after adding the recall code to query.py.
"""

from query import load_fulllog, embed_exchange_to_conversations

exchanges = load_fulllog()
print(f"Backfilling {len(exchanges)} exchanges into conversations collection...")

for i, ex in enumerate(exchanges, 1):
    embed_exchange_to_conversations(ex)
    print(f"  [{i}/{len(exchanges)}] embedded")

print("Done.")