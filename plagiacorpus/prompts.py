"""Prompt templates used for LLM-based query generation (manuscript, Figures 2-4).

The wording, including the original spelling "rewritter", is kept exactly as used
to generate the released dataset.
"""

# ---------------------------------------------------------------------------
# Class 3 - standard paraphrase (Qwen2.5-14B-Instruct, chat format)
# The Wikipedia component used a more concise user instruction than the
# public-domain (PD) component (Figure 2 shows the PD version).
# ---------------------------------------------------------------------------
CLASS3_SYSTEM = {
    "en": "You are a professional rewritter.",
    "ar": "أنت كاتب محترف تقوم بإعادة صياغة النصوص.",
}

CLASS3_USER = {
    "wikipedia": {
        "en": "Rewrite this completely in your own words. Output ONLY the rewritten text:\n{text}",
        "ar": "أعد صياغة هذا النص بكلماتك الخاصة. أخرج النص المعاد صياغته فقط:\n{text}",
    },
    "public_domain": {
        "en": (
            "Rewrite the following text completely in your own words while keeping "
            "the core meaning. Output ONLY the rewritten text:\n{text}"
        ),
        "ar": "أعد صياغة النص التالي بكلماتك الخاصة مع الحفاظ على المعنى الأساسي. أخرج النص المُعاد صياغته فقط:\n{text}",
    },
}

# Sampling settings per component (top-p was only set for the PD component).
CLASS3_SAMPLING = {
    "wikipedia": {"temperature": 0.7},
    "public_domain": {"temperature": 0.7, "top_p": 0.9},
}

# ---------------------------------------------------------------------------
# Class 4 - complex paraphrase (gemma4:31b, single prompt; Figure 3)
# ---------------------------------------------------------------------------
CLASS4_PROMPT = {
    "en": (
        "You are a professional rewritter. Rewrite the following text completely in your "
        "own words while keeping the core meaning. Output ONLY the rewritten text, without "
        "any introductory phrases:\n\n{text}"
    ),
    "ar": "أنت كاتب محترف تقوم بإعادة صياغة النصوص. أعد صياغة النص التالي بكلماتك الخاصة مع الحفاظ على المعنى الأساسي. أخرج النص المُعاد صياغته فقط بدون أي مقدمات:\n\n{text}",
}

# ---------------------------------------------------------------------------
# Class 5 - cross-lingual reuse (gemma4:26b, single prompt; Figure 4)
# Keys are translation directions (source language -> target language).
# ---------------------------------------------------------------------------
CLASS5_PROMPT = {
    "EN->AR": (
        "You are an expert academic translator. \n"
        "Translate the following English text into perfectly natural Modern Standard Arabic, "
        "and subtly paraphrase the structure so it does not look like a literal machine "
        "translation.\n"
        "Output ONLY the final Arabic text. Do not include any conversational filler, notes, "
        "or English words.\n\n"
        "English Text:\n{text}"
    ),
    "AR->EN": (
        "You are an expert academic translator. \n"
        "Translate the following Arabic text into formal, academic English, and subtly "
        "paraphrase the structure so it does not look like a literal machine translation.\n"
        "Output ONLY the final English text. Do not include any conversational filler, notes, "
        "or Arabic words.\n\n"
        "Arabic Text:\n{text}"
    ),
}
