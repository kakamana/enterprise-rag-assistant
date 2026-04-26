"""Synthetic enterprise-policy corpus + Q&A generator.

Produces:
    data/processed/policy_corpus.parquet   — 5,000 paragraphs across 12 domains
    data/processed/policy_qa.parquet       — 1,000 Q&A pairs with ground-truth source IDs

Run as:
    python -m enterprise_rag.data
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW = DATA_DIR / "raw"
PROCESSED = DATA_DIR / "processed"

SEED = 42
N_PARAGRAPHS = 5_000
N_QA = 1_000

DOMAINS = [
    "HR", "Finance", "IT", "Travel", "Procurement", "Compliance",
    "Security", "Legal", "HealthSafety", "Sustainability", "Data", "Vendor",
]

# Per-domain vocabulary that drives both paragraph templating and Q&A generation.
DOMAIN_VOCAB: dict[str, dict[str, list[str]]] = {
    "HR": {
        "subject": ["annual leave", "sick leave", "parental leave", "performance review",
                    "probation", "remote work", "overtime", "bereavement leave",
                    "compensation", "promotion eligibility"],
        "verb": ["request", "approve", "submit", "document", "escalate"],
        "limit": ["30 calendar days", "14 working days", "5 working days",
                  "two pay cycles", "one calendar quarter"],
        "actor": ["the line manager", "the HR Business Partner", "the head of department"],
    },
    "Finance": {
        "subject": ["expense reimbursement", "purchase order", "invoice approval",
                    "petty cash", "corporate card", "year-end accrual",
                    "vendor payment", "travel advance", "tax filing", "budget transfer"],
        "verb": ["approve", "settle", "record", "reconcile", "submit"],
        "limit": ["AED 500", "AED 5,000", "AED 50,000", "USD 1,000", "EUR 250"],
        "actor": ["the Finance Controller", "the CFO office", "Accounts Payable"],
    },
    "IT": {
        "subject": ["password reset", "VPN access", "device provisioning",
                    "software installation", "MFA enrollment", "ticket SLA",
                    "asset return", "shared drive access", "email retention", "patching"],
        "verb": ["request", "configure", "complete", "log", "review"],
        "limit": ["within 24 hours", "within 72 hours", "every 90 days",
                  "every 12 months", "before offboarding"],
        "actor": ["the IT Service Desk", "the system owner", "the Information Security team"],
    },
    "Travel": {
        "subject": ["flight booking", "hotel booking", "per diem",
                    "ground transport", "visa support", "trip approval",
                    "expense claim", "advance request", "trip insurance", "loyalty points"],
        "verb": ["book", "pre-approve", "claim", "settle", "document"],
        "limit": ["AED 800 per night", "economy class", "USD 75 per day",
                  "10 working days in advance", "within 7 days of return"],
        "actor": ["the travel desk", "the line manager", "the Finance team"],
    },
    "Procurement": {
        "subject": ["supplier onboarding", "RFP issuance", "purchase requisition",
                    "single-source justification", "competitive bidding",
                    "contract renewal", "supplier evaluation", "spend threshold",
                    "framework agreement", "SOW approval"],
        "verb": ["initiate", "approve", "document", "review", "renew"],
        "limit": ["AED 25,000", "USD 50,000", "three competitive bids",
                  "two-tier approval", "annual renewal cycle"],
        "actor": ["the Procurement lead", "the Category Manager", "the Legal counsel"],
    },
    "Compliance": {
        "subject": ["AML check", "KYC review", "conflict of interest",
                    "gifts and hospitality", "sanctions screening",
                    "whistleblower report", "annual attestation",
                    "regulatory training", "incident disclosure", "policy exception"],
        "verb": ["disclose", "complete", "review", "escalate", "log"],
        "limit": ["within 5 working days", "annually", "every six months",
                  "before any vendor engagement", "prior to year-end"],
        "actor": ["the Compliance Officer", "the Legal team", "the Audit committee"],
    },
    "Security": {
        "subject": ["access badge", "incident response", "tailgating",
                    "clean-desk policy", "visitor escort", "CCTV footage request",
                    "key management", "secure courier", "security clearance",
                    "physical entry log"],
        "verb": ["report", "log", "renew", "escort", "audit"],
        "limit": ["within 24 hours", "monthly", "before site entry",
                  "for the duration of the visit", "every 12 months"],
        "actor": ["the Security Operations Centre", "the site security officer",
                  "the Chief Information Security Officer"],
    },
    "Legal": {
        "subject": ["NDA execution", "contract review", "intellectual property",
                    "GDPR notice", "subpoena handling", "litigation hold",
                    "data processing agreement", "external counsel engagement",
                    "press statement", "regulatory filing"],
        "verb": ["sign", "review", "preserve", "engage", "file"],
        "limit": ["before sharing confidential information",
                  "within 10 working days", "for the statutory retention period",
                  "before publication", "by the regulatory deadline"],
        "actor": ["the General Counsel", "external counsel", "the DPO"],
    },
    "HealthSafety": {
        "subject": ["near-miss report", "PPE compliance", "ergonomic assessment",
                    "fire drill", "first-aid kit", "emergency evacuation",
                    "workplace injury", "site safety induction",
                    "machinery guarding", "hazardous materials handling"],
        "verb": ["report", "complete", "perform", "log", "schedule"],
        "limit": ["within 24 hours", "every quarter", "before site work begins",
                  "annually", "after every incident"],
        "actor": ["the EHS officer", "the floor warden", "the site supervisor"],
    },
    "Sustainability": {
        "subject": ["carbon reporting", "waste segregation", "energy monitoring",
                    "supplier ESG questionnaire", "single-use plastics",
                    "paper consumption", "water usage", "fleet emissions",
                    "renewable procurement", "ESG disclosure"],
        "verb": ["report", "track", "audit", "publish", "reduce"],
        "limit": ["quarterly", "annually", "by the financial year end",
                  "for all sites above 100 FTE", "in the sustainability report"],
        "actor": ["the Sustainability lead", "Facilities", "the ESG committee"],
    },
    "Data": {
        "subject": ["data classification", "personal data export", "data retention",
                    "access review", "data subject request", "data lineage",
                    "data quality incident", "cross-border transfer",
                    "encryption at rest", "data sharing agreement"],
        "verb": ["classify", "log", "approve", "review", "purge"],
        "limit": ["within 30 days", "every 12 months", "before any export",
                  "for 7 years", "prior to publication"],
        "actor": ["the Data Steward", "the DPO", "the Data Protection Officer"],
    },
    "Vendor": {
        "subject": ["vendor risk assessment", "vendor onboarding", "SLA review",
                    "vendor offboarding", "third-party access", "vendor audit",
                    "scope change", "vendor escalation", "vendor performance review",
                    "vendor security questionnaire"],
        "verb": ["complete", "review", "renew", "escalate", "document"],
        "limit": ["before contract signature", "annually",
                  "within the SLA window", "before granting system access",
                  "every six months"],
        "actor": ["the Vendor Manager", "Procurement", "Information Security"],
    },
}


SENTENCE_TEMPLATES = [
    "{actor} must {verb} every {subject} within {limit}.",
    "All {subject} cases require approval by {actor} {limit}.",
    "Employees should {verb} a {subject} request {limit} via the official portal.",
    "The standard limit for {subject} is {limit}, after which {actor} review is required.",
    "Failure to {verb} a {subject} record {limit} may result in escalation by {actor}.",
    "The {subject} workflow is owned by {actor} and is reviewed {limit}.",
    "Exceptions to the {subject} policy must be documented by {actor} {limit}.",
    "Any {subject} that exceeds {limit} requires written sign-off from {actor}.",
]

QUESTION_TEMPLATES = [
    "What is the limit for {subject}?",
    "Who approves a {subject}?",
    "How quickly must I {verb} a {subject}?",
    "What is the policy on {subject}?",
    "When does {actor} review {subject}?",
    "What happens if I exceed the {subject} limit?",
    "Is there an exception process for {subject}?",
]

ANSWER_TEMPLATES = [
    "The limit for {subject} is {limit}; approval sits with {actor}.",
    "{actor} approves every {subject}; the standard limit is {limit}.",
    "You must {verb} a {subject} {limit}; escalation goes to {actor}.",
    "Policy requires {subject} to be handled {limit} by {actor}.",
    "Exceptions to {subject} require sign-off from {actor} {limit}.",
]


def _make_paragraph(rng: np.random.Generator, domain: str) -> str:
    vocab = DOMAIN_VOCAB[domain]
    n_sent = int(rng.integers(2, 5))
    parts: list[str] = []
    for _ in range(n_sent):
        tmpl = SENTENCE_TEMPLATES[int(rng.integers(0, len(SENTENCE_TEMPLATES)))]
        parts.append(tmpl.format(
            actor=rng.choice(vocab["actor"]),
            verb=rng.choice(vocab["verb"]),
            subject=rng.choice(vocab["subject"]),
            limit=rng.choice(vocab["limit"]),
        ))
    return " ".join(parts)


def build_corpus(n_paragraphs: int = N_PARAGRAPHS, seed: int = SEED) -> pd.DataFrame:
    """Generate `n_paragraphs` policy paragraphs balanced across the 12 domains."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    per_domain = n_paragraphs // len(DOMAINS)
    extra = n_paragraphs - per_domain * len(DOMAINS)
    for di, domain in enumerate(DOMAINS):
        count = per_domain + (1 if di < extra else 0)
        # Group paragraphs into "documents" of 6-10 paragraphs.
        i = 0
        doc_seq = 1
        while i < count:
            doc_len = int(rng.integers(6, 11))
            doc_id = f"{domain}-{doc_seq:03d}"
            for pid in range(doc_len):
                if i >= count:
                    break
                rows.append({
                    "doc_id": doc_id,
                    "domain": domain,
                    "paragraph_id": pid,
                    "text": _make_paragraph(rng, domain),
                })
                i += 1
            doc_seq += 1
    df = pd.DataFrame(rows)
    return df


def build_qa(corpus: pd.DataFrame, n_qa: int = N_QA, seed: int = SEED + 1) -> pd.DataFrame:
    """Generate Q&A pairs with explicit ground-truth source IDs.

    Each question is anchored to a specific (domain, subject) tuple and the
    `source_doc_ids` field links back to the paragraph(s) that mention that subject.
    """
    rng = np.random.default_rng(seed)
    # Pre-index the corpus: for each (domain, subject token) collect candidate paragraphs.
    by_domain_subject: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for _, row in corpus.iterrows():
        d = row["domain"]
        text_lower = row["text"].lower()
        for subj in DOMAIN_VOCAB[d]["subject"]:
            if subj.lower() in text_lower:
                by_domain_subject.setdefault((d, subj), []).append(
                    (row["doc_id"], int(row["paragraph_id"]))
                )

    pairs = list(by_domain_subject.keys())
    rng.shuffle(pairs)

    rows: list[dict] = []
    qid = 0
    while len(rows) < n_qa:
        for domain, subject in pairs:
            if len(rows) >= n_qa:
                break
            candidates = by_domain_subject[(domain, subject)]
            if not candidates:
                continue
            vocab = DOMAIN_VOCAB[domain]
            actor = str(rng.choice(vocab["actor"]))
            verb = str(rng.choice(vocab["verb"]))
            limit = str(rng.choice(vocab["limit"]))
            qtmpl = QUESTION_TEMPLATES[int(rng.integers(0, len(QUESTION_TEMPLATES)))]
            atmpl = ANSWER_TEMPLATES[int(rng.integers(0, len(ANSWER_TEMPLATES)))]
            question = qtmpl.format(subject=subject, verb=verb, actor=actor)
            answer = atmpl.format(subject=subject, verb=verb, actor=actor, limit=limit)
            n_sources = 1 if rng.random() < 0.7 else min(2, len(candidates))
            picks = rng.choice(len(candidates), size=n_sources, replace=False)
            sources = ";".join(f"{candidates[i][0]}:{candidates[i][1]}" for i in picks)
            rows.append({
                "qa_id": f"Q{qid:05d}",
                "domain": domain,
                "question": question,
                "answer": answer,
                "source_doc_ids": sources,
            })
            qid += 1
    return pd.DataFrame(rows)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    corpus = build_corpus()
    corpus.to_parquet(PROCESSED / "policy_corpus.parquet", index=False)

    qa = build_qa(corpus)
    qa.to_parquet(PROCESSED / "policy_qa.parquet", index=False)

    print(
        f"wrote {len(corpus):,} paragraphs -> data/processed/policy_corpus.parquet\n"
        f"wrote {len(qa):,} Q&A pairs    -> data/processed/policy_qa.parquet"
    )


if __name__ == "__main__":
    main()
