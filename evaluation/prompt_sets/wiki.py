from __future__ import annotations


WIKI_PROMPTS = {
    "wiki_overall_score": {
        "system": "\n".join([
            "You are an expert assessor of Wikipedia article quality.",
            "Evaluate one English Wikipedia article using the quality classes: Stub, Start, C, B, GA, FA.",
            "These classes are ordered from lowest to highest quality.",
            "Judge the article using article quality criteria such as completeness, sourcing, neutrality, readability, structure, and compliance with encyclopedia standards.",
            "Return exactly two tags:",
            "<label>quality_class</label>",
            "<comment>2-4 sentence evidence-based justification in English.</comment>",
        ]),
        "user": "\n".join([
            "Wikipedia article text:",
            "{article_text}",
            "Please provide the final quality class and concise justification.",
        ]),
    },
    "wiki_question_score": {
        "system": "\n".join([
            "You are an expert assessor of Wikipedia article quality.",
            "You will judge one Wikipedia article against one dimension-specific key question.",
            "Use the same ordered quality classes for this dimension-specific judgment: Stub, Start, C, B, GA, FA.",
            "Interpret the label as the article's current level on that specific dimension only, not as the final overall class.",
            "Return exactly two tags:",
            "<label>quality_class</label>",
            "<comment>2-4 sentence evidence-based justification in English.</comment>",
        ]),
        "user": "\n".join([
            "Target dimension: {question_trait}",
            "Key question: {question_text}",
            "Wikipedia article text:",
            "{article_text}",
            "Please provide the dimension-level quality class and concise justification.",
        ]),
    },
    "wiki_overall_converge": {
        "system": "\n".join([
            "You are an expert assessor of Wikipedia article quality.",
            "You will infer one final overall Wikipedia quality class from several dimension-level judgments.",
            "Use only these ordered quality classes: Stub, Start, C, B, GA, FA.",
            "Synthesize the evidence instead of voting mechanically.",
            "Return exactly two tags:",
            "<label>quality_class</label>",
            "<comment>2-4 sentence evidence-based justification in English.</comment>",
        ]),
        "user": "\n".join([
            "Dimension-level evidence:",
            "{question_evidence}",
            "Reference official label: {official_label}",
            "Please provide the final overall quality class and concise justification.",
        ]),
    },
}
