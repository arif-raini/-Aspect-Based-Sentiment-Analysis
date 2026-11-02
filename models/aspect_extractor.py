"""
Aspect Extractor Module for Aspect-Based Sentiment Analysis (ABSA).
Combines spaCy dependency parsing, noun phrase chunking, NLTK POS patterns,
and a domain aspect taxonomy to extract product aspects and their localized context.
"""

import re
import spacy
from config import ASPECT_CATEGORIES, KEYWORD_TO_ASPECT


class AspectExtractor:
    """
    Extracts product aspects and their localized sentiment-bearing context
    using dependency parsing, noun chunks, and taxonomy mapping.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
        except Exception:
            self.nlp = None

        # Sort taxonomy keywords by length (longest/most specific first)
        self.sorted_keywords = sorted(KEYWORD_TO_ASPECT.keys(), key=lambda k: len(k.split()), reverse=True)

    def segment_clauses(self, text: str) -> list[str]:
        """
        Segment compound and complex sentences into individual clauses
        using conjunctions, semicolons, and punctuation boundaries.
        """
        pattern = r'[,;.]|\b(?:but|however|although|though|yet|while|whereas|and)\b'
        clauses = re.split(pattern, text, flags=re.IGNORECASE)
        cleaned_clauses = [c.strip() for c in clauses if c and len(c.strip()) > 2]
        return cleaned_clauses if cleaned_clauses else [text.strip()]

    def _match_taxonomy(self, text: str) -> tuple[str | None, str | None]:
        """
        Matches a word or phrase against the domain aspect taxonomy.
        Returns (canonical_aspect, matched_keyword).
        Checks multi-word phrases first to prioritize specific compound aspects.
        """
        text_lower = text.lower().strip()
        for kw in self.sorted_keywords:
            # Word boundary regex match
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, text_lower):
                return KEYWORD_TO_ASPECT[kw], kw

        return None, None

    def extract_aspect_contexts(self, text: str) -> list[dict]:
        """
        Core ABSA Function:
        Extracts all identified aspects along with their specific isolated
        clause/context window. This ensures each aspect is classified independently.
        """
        if not text or not isinstance(text, str):
            return []

        clauses = self.segment_clauses(text)
        found_aspects = {}
        matched_spans = set()

        # 1. Match multi-word and single-word keywords across clauses (longest first)
        for clause in clauses:
            clause_lower = clause.lower()
            for kw in self.sorted_keywords:
                pattern = r'\b' + re.escape(kw) + r'\b'
                match = re.search(pattern, clause_lower)
                if match:
                    category = KEYWORD_TO_ASPECT[kw]
                    # Check if this keyword is a sub-phrase of an already matched longer keyword in this clause
                    span_key = (clause, match.start(), match.end())
                    if not any(clause == m_clause and match.start() >= m_start and match.end() <= m_end for m_clause, m_start, m_end in matched_spans):
                        matched_spans.add(span_key)
                        if category not in found_aspects:
                            found_aspects[category] = {
                                "aspect": category,
                                "raw_term": kw,
                                "contexts": [clause]
                            }
                        else:
                            if clause not in found_aspects[category]["contexts"]:
                                found_aspects[category]["contexts"].append(clause)

        # 2. Dependency-based parsing if spaCy is available to capture opinion modifiers
        if self.nlp:
            try:
                doc = self.nlp(text)
                for chunk in doc.noun_chunks:
                    chunk_lower = chunk.text.lower().strip()
                    category, kw = self._match_taxonomy(chunk_lower)
                    if category:
                        # Extract connected adjectives, predicates, negations
                        modifiers = []
                        root = chunk.root
                        for child in root.children:
                            if child.dep_ in ("amod", "advmod", "neg", "acomp"):
                                modifiers.append(child.text)
                        
                        if root.dep_ in ("nsubj", "dobj") and root.head.pos_ in ("VERB", "AUX"):
                            modifiers.append(root.head.text)
                            for sib in root.head.children:
                                if sib != root and sib.dep_ in ("acomp", "advmod", "neg", "attr"):
                                    modifiers.append(sib.text)

                        if modifiers and category in found_aspects:
                            phrase = f"{' '.join(modifiers)} {chunk_lower}".strip()
                            if phrase not in found_aspects[category]["contexts"]:
                                found_aspects[category]["contexts"].append(phrase)
            except Exception:
                pass

        # 3. Consolidate context for each aspect
        final_results = []
        for category, data in found_aspects.items():
            # Use the most informative clause as primary context
            primary_context = data["contexts"][0]
            final_results.append({
                "aspect": category,
                "raw_term": data["raw_term"],
                "context": primary_context if primary_context else text,
                "all_clauses": data["contexts"]
            })

        return final_results
