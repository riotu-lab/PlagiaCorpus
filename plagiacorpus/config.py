"""Constants shared across the PlagiaCorpus pipeline.

Model names, record labels, provenance strings and file names live here so that
every script writes exactly the same values into the dataset.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Generation models (manuscript, Table 3)
# ---------------------------------------------------------------------------
CLASS3_MODEL = "Qwen/Qwen2.5-14B-Instruct"  # Hugging Face Transformers
CLASS4_MODEL = "gemma4:31b"                  # Ollama
CLASS5_MODEL = "gemma4:26b"                  # Ollama

# ---------------------------------------------------------------------------
# Record-level labels (manuscript, Table 6)
# ---------------------------------------------------------------------------
GEN_DIRECT_COPY = "None (Direct Human Copy)"
GEN_ALGORITHMIC = "None (Algorithmic Modification)"
GEN_HUMAN = "Human (Original Text)"
UNKNOWN_ID = "UNKNOWN_ID"  # original_text_id of Safe (negative) queries

# ---------------------------------------------------------------------------
# Source components and their identifiers
# ---------------------------------------------------------------------------
COMPONENTS = {
    "wikipedia": {"prefix": "WIKI", "domain": "Wikimedia Foundation (2024 Dumps)"},
    "public_domain": {"prefix": "PD", "domain": "Project Gutenberg / Internet Archive"},
}

PD_DATASETS = {
    "en": "PleIAs/English-PD",
    "ar": "mohres/The_Arabic_E-Book_Corpus",
}

# ---------------------------------------------------------------------------
# Default file layout
# ---------------------------------------------------------------------------
INTERIM_DIR = Path("data/interim")
FINAL_DIR = Path("data/final")
RESULTS_DIR = Path("results")

WIKI_COMPONENT_FILE = INTERIM_DIR / "wikipedia_component.json"
PD_COMPONENT_FILE = INTERIM_DIR / "public_domain_component.json"
BASE_QUERIES_FILE = INTERIM_DIR / "queries_class1-3_safe.json"
CLASS4_DIR = INTERIM_DIR / "class4"
CLASS5_FILE = INTERIM_DIR / "queries_class5.json"

CORPUS_FILE = FINAL_DIR / "PlagiaCorpus_corpus.json"
TRAIN_FILE = FINAL_DIR / "PlagiaCorpus_queries_train.json"
TEST_FILE = FINAL_DIR / "PlagiaCorpus_queries_test.json"

# ---------------------------------------------------------------------------
# Wikipedia crawl topics (Source Corpus Construction)
# ---------------------------------------------------------------------------
WIKI_TOPICS = {
    "en": [
        "Literature", "History of Saudi Arabia", "Philosophy", "Copyright Law",
        "Artificial Intelligence", "Poetry", "Middle Eastern Culture", "Science", "Law",
        "Classic Literature", "Modernism", "Post-colonial Literature", "Renaissance Art",
        "Creative Writing", "Epic Poetry", "Literary Criticism", "Drama", "World Cinema",
        "Islamic Golden Age", "Ottoman Empire", "Ancient Egypt", "Silk Road",
        "Arabian Peninsula", "Cultural Heritage", "Archaeology", "Anthropology",
        "Intellectual Property", "Digital Rights Management", "Plagiarism", "Fair Use",
        "International Law", "Contract Law", "Legal Ethics", "Patent Law",
        "Machine Learning", "Quantum Physics", "Astronomy", "Microbiology", "Data Science",
        "Robotics", "Cybersecurity", "Blockchain", "Sociology", "Economics", "Human Rights",
        "Political Science", "Environmental Science", "Geography", "Urbanization",
        "Globalization", "Ethics", "Arabic Calligraphy", "Islamic Architecture",
        "Vision 2030", "Gulf Cooperation Council", "History of Riyadh", "Najd", "Hejaz",
        "Arabic Music", "Al-Ula",
    ],
    "ar": [
        "أدب", "تاريخ السعودية", "فلسفة", "حقوق التأليف والنشر", "ذكاء اصطناعي", "شعر",
        "ثقافة الشرق الأوسط", "علوم", "قانون", "الأدب العربي", "الشعر الجاهلي",
        "الرواية العربية", "أدب المهجر", "فن الخط العربي", "المعلقات", "النقد الأدبي",
        "المسرح العربي", "السينما العربية", "الأدب الأندلسي", "تاريخ الإسلام",
        "الدولة السعودية الأولى",
        "تاريخ الإسلام",  # listed twice in the original run; kept for fidelity
        "العصر العباسي", "العصر الأموي", "تراث شعبي", "آثار الجزيرة العربية", "طريق البخور",
        "تاريخ مكة", "تاريخ المدينة المنورة", "الملكية الفكرية", "قانون العمل",
        "القرصنة الرقمية", "الأخلاقيات المهنية", "القانون الدولي",
        "أنظمة المملكة العربية السعودية", "حماية البيانات", "جرائم معلوماتية", "تعلم الآلة",
        "فيزياء الكم", "علم الفلك", "تقنية النانو", "الأمن السيبراني", "علم البيانات",
        "الروبوتات", "هندسة البرمجيات", "الطاقة المتجددة", "فلسفة عربية", "علم الاجتماع",
        "اقتصاد كلي", "علوم سياسية", "جغرافيا العالم العربي", "عولمة", "رؤية السعودية 2030",
        "تنمية مستدامة", "حوار الثقافات", "فلسفة الأخلاق", "العمارة الإسلامية",
        "تاريخ الرياض", "منطقة نجد", "منطقة الحجاز", "تراث العلا", "الدرعية",
        "موسيقى عربية", "سوق عكاظ", "مجلس التعاون الخليجي",
    ],
}

# Topics for the independent Safe (negative) control pools
WIKI_SAFE_TOPICS = {
    "en": ["Mathematics", "Physics", "Chemistry"],
    "ar": ["رياضيات", "فيزياء", "كيمياء"],
}
