"""
Batch runner — feeds a list of questions through the real query_knowledge pipeline
so you don't have to type each one manually. Useful for building up exchange volume
to test Layer 3 distillation.
"""

import time
from query import query_knowledge, conversation_history, ledger

OPTION_C_QUESTIONS = [
    "I think alpha generation strategies are fundamentally unsustainable long term because they get arbitraged away once enough firms copy them — do you agree based on what's in the text?",
    "My take is that Vanguard's operational efficiency approach is the smartest long-term bet, not because it's flashy but because it's defensible against competition. What does the text suggest about this?",
    "I find it interesting that Ray Dalio basically tried to codify Stoic-style discipline into an algorithm with PrinciplesOS. Do you see a real connection between Stoicism and his investment principles?",
    "Personally I don't think raw alpha-chasing reflects good judgment under uncertainty — it feels more like Marcus Aurelius would caution against it. Does the text support that comparison?",
    "I prefer comparative analysis over narrative explanations when I'm trying to evaluate firms against each other — can you lay out Bridgewater, BlackRock, and Vanguard side by side on AI strategy?",
    "When I think about resilience, I care less about avoiding setbacks and more about how systems recover from arbitrage or market shifts. How does Marcus Aurelius frame resilience, and does it map to that?",
    "I want to understand the actual mechanism of how JPMorgan's Contract Intelligence works, not just that it exists — can you go deep on the technical side rather than summarizing?",
    "My instinct is that most 'AI strategy' framing in finance is just marketing for what's really applied statistics. Does anything in the text push back on that view?",

    "I think BlackRock's risk-first approach with Aladdin is actually more defensible than Bridgewater's alpha-chasing, because risk management compounds while alpha decays — does the text support that?",
    "I'm skeptical that any of these three firms have a real moat in AI specifically, since the underlying techniques aren't proprietary. What does the text suggest about defensibility?",
    "When I read Marcus Aurelius, I notice he keeps returning to the idea of accepting what you can't control — I think that's actually a risk management philosophy before risk management existed. Agree?",
    "I don't just want the facts on JPMorgan's AI spend, I want your read on whether that spend level signals genuine strategic conviction or just keeping up with competitors.",
    "My working theory is that operational efficiency AI use cases (like Vanguard's) are underrated compared to flashy trading AI, because they're boring and compound quietly. Does the text back this up at all?",
    "I tend to distrust strategies that sound impressive in a press release — PrinciplesOS sounds impressive. Does the text give any indication of actual measurable results, or is it mostly narrative?",
    "Comparing Stoic restraint to disciplined investing feels like a stretch to some people, but I think the parallel is real — what specifically in the text would support or undercut that?",
    "I like to push back on consensus views — what's the strongest argument against the idea that alpha generation gets arbitraged away?",
    "Give me the data-first version, not the narrative version: what are the hard numbers mentioned for each firm's AUM and AI spend?",
    "I think most people misunderstand what 'AI strategy' means in asset management — they assume it's all trading bots when a lot of it is operational. Does the text support that distinction?",
    "How would a Stoic philosopher critique Bridgewater's approach of constantly trying to beat the market rather than accepting market outcomes?",
    "I want a blunt, no-hedging answer here: which of the three firms is making the smartest long-term AI bet, and why?",
    "What's the throughline between Marcus Aurelius's view on ambition and how these finance firms talk about competitive advantage?",
    "I prefer when answers state a clear position rather than listing both sides — based on the text, what's the single biggest risk to Bridgewater's strategy specifically?",
    "Does the text suggest JPMorgan views AI as a cost center or a genuine competitive weapon? I want your interpretation, not just a restatement.",
    "I think Vanguard's approach is underappreciated because it doesn't generate headlines the way alpha generation does — what evidence in the text either supports or weakens that take?",
    "Final one: synthesizing everything discussed, do you think Stoic philosophy and modern risk management are actually the same idea wearing different clothes, or is that a stretch?",
]

QUESTIONS = [
    # Finance + opinion
    "What is Bridgewater's AI strategy?",
    "Which of these three firms would you bet on long term?",
    "Why do you think alpha generation gets arbitraged away?",
    "What would Ray Dalio think about Vanguard's approach?",
    "How does BlackRock's Aladdin system actually work?",
    "Who founded Bridgewater?",

    # Philosophy + application
    "What does Marcus Aurelius say about dealing with failure?",
    "How would you apply that to a startup founder?",
    "Which Stoic principle is most relevant to investing?",
    "Do you think Dalio's principles are Stoic in nature?",
    "What does Marcus Aurelius say about anger?",
    "How should one respond to criticism according to the text?",

    # Cross-domain synthesis
    "What's the relationship between Stoicism and risk management?",
    "How does BlackRock's approach reflect any philosophical framework?",
    "If Marcus Aurelius ran a hedge fund, what would his strategy be?",
    "Is alpha generation compatible with a Stoic mindset?",

    # JPMorgan + tech depth
    "How does JPMorgan use AI across its business?",
    "What does JPMorgan spend on technology annually?",
    "How does Contract Intelligence work?",
    "Compare JPMorgan's AI use to Bridgewater's.",

    "Which firm manages the most assets overall?",
    "What's the biggest risk across all three asset managers?",
    "Is Vanguard's approach more sustainable than Bridgewater's?",
    "What's the core difference between Stoic acceptance and giving up?",
    "Would you recommend a finance professional read Marcus Aurelius?",
]

if __name__ == "__main__":
    print(f"Running {len(OPTION_C_QUESTIONS)} questions through the pipeline...\n")
    for i, q in enumerate(OPTION_C_QUESTIONS, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(OPTION_C_QUESTIONS)}] {q}")
        print('='*60)
        query_knowledge(q, conversation_history)
        time.sleep(1)  # small pause between calls

    ledger.session_summary()