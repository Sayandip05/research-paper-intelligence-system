"""
Intent Classifier for Section-Aware Retrieval

Rule-based, deterministic intent detection.
NO ML, NO embeddings, NO LLM.

Maps user queries to allowed section filters.
"""

from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class IntentResult:
    """Result of intent classification"""
    intent: str
    allowed_sections: List[str]
    confidence: float  # 1.0 for exact match, 0.5 for fallback


# Intent → Allowed Sections Mapping (STRICT)
INTENT_SECTION_MAP = {
    "summary": ["Abstract", "Introduction"],
    "methodology": ["Methods"],
    "experiments": ["Experiments", "Results"],
    "results": ["Results"],
    "research_gaps": ["Discussion", "Limitations", "Future Work"],
    "limitations": ["Discussion", "Limitations"],
    "future_work": ["Future Work"],
    "comparison": ["Results", "Experiments"],
    "citation": ["References"],
    "general": ["Abstract", "Introduction", "Methods", "Results"],
}

# Keyword → Intent Mapping (order matters - more specific first)
INTENT_KEYWORDS = {
    "limitations": [
        "limitation", "drawback", "shortcoming", "weakness", "problem with",
        "issue with", "challenge", "constraint", "restriction", "downside"
    ],
    "future_work": [
        "future work", "future direction", "next step", "improve", "extension",
        "further research", "open question", "remaining"
    ],
    "research_gaps": [
        "gap", "missing", "lack", "unexplored", "underexplored", "overlooked",
        "not addressed", "unresolved"
    ],
    "methodology": [
        "method", "approach", "technique", "algorithm", "how does", "how do",
        "procedure", "framework", "architecture", "design", "implementation"
    ],
    "experiments": [
        "experiment", "evaluation", "benchmark", "test", "dataset", "baseline",
        "ablation", "hyperparameter", "training", "fine-tun"
    ],
    "results": [
        "result", "performance", "accuracy", "score", "metric", "outcome",
        "finding", "achieve", "outperform"
    ],
    "comparison": [
        "compare", "comparison", "versus", "vs", "differ", "better than",
        "worse than", "relative to", "against"
    ],
    "summary": [
        "summary", "summarize", "overview", "what is", "explain", "describe",
        "introduction", "main idea", "key point", "tldr", "gist"
    ],
    "citation": [
        "reference", "cite", "citation", "source", "bibliography", "paper by"
    ],
}

# Intent Priority (Higher = More Specific, Wins Conflicts)
# When multiple intents match, the highest priority is selected
INTENT_PRIORITY = {
    "citation": 100,
    "limitations": 90,
    "future_work": 85,
    "research_gaps": 80,
    "methodology": 70,
    "experiments": 60,
    "results": 50,
    "comparison": 40,
    "summary": 20,
    "general": 10,
}


class IntentClassifier:
    """
    Deterministic, keyword-based intent classifier.
    
    NO ML, NO embeddings, NO LLM.
    
    Uses priority-based resolution when multiple intents match:
    - More specific intents (methodology) beat generic ones (summary)
    - Priority order: citation > limitations > methodology > ... > general
    
    Usage:
        classifier = IntentClassifier()
        result = classifier.classify("What are the limitations of LoRA?")
        # result.intent = "limitations"
        # result.allowed_sections = ["Discussion", "Limitations"]
    """
    
    def __init__(self):
        self.intent_keywords = INTENT_KEYWORDS
        self.intent_section_map = INTENT_SECTION_MAP
        self.intent_priority = INTENT_PRIORITY
    
    def classify(self, query: str) -> IntentResult:
        """
        Classify user query into an intent with allowed sections.
        
        Uses PRIORITY-BASED resolution:
        1. Find all matching intents (keyword matches)
        2. Select the intent with HIGHEST priority
        
        Args:
            query: User's natural language question
            
        Returns:
            IntentResult with intent name and allowed sections
        """
        query_lower = query.lower()
        
        # Collect ALL matching intents (not just first match)
        matched_intents = set()
        
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    matched_intents.add(intent)
                    break  # One keyword match is enough for this intent
        
        if matched_intents:
            # Select intent with HIGHEST priority
            best_intent = max(matched_intents, key=lambda i: self.intent_priority.get(i, 0))
            
            return IntentResult(
                intent=best_intent,
                allowed_sections=self.intent_section_map[best_intent],
                confidence=1.0
            )
        
        # No match - fallback to general
        return IntentResult(
            intent="general",
            allowed_sections=self.intent_section_map["general"],
            confidence=0.5
        )
    
    def get_qdrant_filter(self, intent_result: IntentResult) -> dict:
        """
        Build Qdrant filter from intent result.
        
        Excludes:
        - "Unknown" (always)
        - "References" (unless intent == citation)
        
        Args:
            intent_result: Result from classify()
            
        Returns:
            Qdrant filter dict for query_points()
        """
        allowed = intent_result.allowed_sections.copy()
        
        # Never include Unknown
        if "Unknown" in allowed:
            allowed.remove("Unknown")
        
        # Only include References for citation intent
        if intent_result.intent != "citation" and "References" in allowed:
            allowed.remove("References")
        
        # Build Qdrant filter
        qdrant_filter = {
            "must": [
                {
                    "key": "section_title",
                    "match": {
                        "any": allowed
                    }
                }
            ]
        }
        
        return qdrant_filter


# Singleton instance
_classifier = None

def get_intent_classifier() -> IntentClassifier:
    """Get singleton IntentClassifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier


if __name__ == "__main__":
    # Test the classifier
    classifier = IntentClassifier()
    
    test_queries = [
        "What are the limitations of LoRA?",
        "How does QLoRA work?",
        "Compare LoRA to full fine-tuning",
        "Summarize the paper",
        "What experiments were conducted?",
        "What are the results?",
        "Tell me about future work",
        "What references does the paper cite?",
        "Explain the methodology",
        "Random question about nothing specific",
    ]
    
    print("\n" + "="*70)
    print("  INTENT CLASSIFIER TEST")
    print("="*70)
    
    for query in test_queries:
        result = classifier.classify(query)
        qdrant_filter = classifier.get_qdrant_filter(result)
        
        print(f"\n  Q: {query}")
        print(f"  → Intent: {result.intent} (conf: {result.confidence})")
        print(f"  → Sections: {result.allowed_sections}")
