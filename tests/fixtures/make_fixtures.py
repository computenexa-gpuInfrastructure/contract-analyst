"""Writes the tiny hand-made fixtures in the real CUAD / ContractNLI formats (invented contracts, not dataset
copies). Run `python tests/fixtures/make_fixtures.py` after changing the texts below."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent

ACME = """MASTER SUPPLY AGREEMENT

This Master Supply Agreement (the "Agreement") is made on March 3, 2021 between Acme Widgets, Inc. ("Supplier") and Blue River Foods LLC ("Buyer").

1. Term. This Agreement starts on April 1, 2021 (the "Effective Date") and ends on March 31, 2024. It renews automatically for successive one (1) year terms unless either party gives sixty (60) days' written notice of non-renewal.

2. Exclusivity. Buyer shall purchase the Products exclusively from Supplier during the Term.

3. Liability. In no event shall either party's total liability exceed the fees paid in the twelve months before the claim.

4. Assignment. Neither party may assign this Agreement without the prior written consent of the other party.

5. Governing Law. This Agreement is governed by the laws of the State of New York.
"""

NORTHWIND = """SERVICES AGREEMENT

This Services Agreement is dated 5/6/2019 between Northwind Ltd. ("Provider") and Contoso Corp. ("Client").

Either party may terminate this Agreement for convenience on thirty (30) days' notice.

Provider shall maintain commercial general liability insurance of at least $1,000,000.

This Agreement is governed by the laws of Ontario, Canada.
"""

CUAD_ANSWERS = {  # category -> quotes (must be in the text)
    "AcmeWidgets_MSA": {
        "Document Name": ["MASTER SUPPLY AGREEMENT"],
        "Parties": ["Acme Widgets, Inc.", "Blue River Foods LLC"],
        "Agreement Date": ["March 3, 2021"],
        "Effective Date": ["This Agreement starts on April 1, 2021"],
        "Expiration Date": ["ends on March 31, 2024"],
        "Renewal Term": ["It renews automatically for successive one (1) year terms"],
        "Notice Period To Terminate Renewal": ["unless either party gives sixty (60) days' written notice of "
                                               "non-renewal"],
        "Exclusivity": ["Buyer shall purchase the Products exclusively from Supplier during the Term."],
        "Cap On Liability": ["In no event shall either party's total liability exceed the fees paid in the twelve "
                             "months before the claim."],
        "Anti-Assignment": ["Neither party may assign this Agreement without the prior written consent of the "
                            "other party."],
        "Governing Law": ["This Agreement is governed by the laws of the State of New York."],
    },
    "Northwind_Services": {
        "Parties": ["Northwind Ltd.", "Contoso Corp."],
        "Agreement Date": ["5/6/2019"],
        "Termination For Convenience": ["Either party may terminate this Agreement for convenience on thirty (30) "
                                        "days' notice."],
        "Insurance": ["Provider shall maintain commercial general liability insurance of at least $1,000,000."],
        "Governing Law": ["This Agreement is governed by the laws of Ontario, Canada."],
    },
}
ALL_CATEGORIES = ["Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date", "Renewal Term",
                  "Notice Period To Terminate Renewal", "Governing Law", "Termination For Convenience",
                  "Cap On Liability", "Uncapped Liability", "Anti-Assignment", "Audit Rights", "Exclusivity",
                  "Insurance", "Minimum Commitment", "Ip Ownership Assignment", "Change Of Control", "Non-Compete",
                  "Liquidated Damages"]
MASTER = [
    {"Filename": "AcmeWidgets_MSA.pdf", "Parties-Answer": 'Acme Widgets, Inc. ("Supplier"); Blue River Foods LLC '
     '("Buyer")', "Agreement Date-Answer": "3/3/21", "Effective Date-Answer": "4/1/21",
     "Expiration Date-Answer": "3/31/24", "Renewal Term-Answer": "successive 1 year",
     "Notice Period To Terminate Renewal- Answer": "60 days", "Governing Law-Answer": "New York"},
    {"Filename": "NORTHWIND_SERVICES.PDF'", "Parties-Answer": 'Northwind Ltd. ("Provider"); Contoso Corp. '
     '("Client")', "Agreement Date-Answer": "5/6/19", "Effective Date-Answer": "", "Expiration Date-Answer": "",
     "Renewal Term-Answer": "", "Notice Period To Terminate Renewal- Answer": "",
     "Governing Law-Answer": "Ontario, Canada"},
]

NDA_1 = ("MUTUAL NON-DISCLOSURE AGREEMENT\n"
         "Confidential Information includes information disclosed orally or in writing.\n"
         "The Recipient shall use Confidential Information solely for the purpose of evaluating the deal.\n"
         "The Recipient may disclose Confidential Information to its employees who need to know it.\n"
         "The obligations in this Agreement survive its termination for three years.\n")
NDA_2 = ("NON-DISCLOSURE AGREEMENT\n"
         "Only information marked as confidential is Confidential Information.\n"
         "The Recipient shall not reverse engineer any samples.\n")
NLI_IDS = ["nda-1", "nda-2", "nda-3", "nda-4", "nda-5", "nda-7", "nda-8", "nda-10", "nda-11", "nda-12", "nda-13",
           "nda-15", "nda-16", "nda-17", "nda-18", "nda-19", "nda-20"]


def sentence_spans(text):
    spans, start = [], 0
    for line in text.splitlines(keepends=True):
        spans.append([start, start + len(line.rstrip("\n"))])
        start += len(line)
    return spans


def nli_doc(doc_id, text, labels):
    spans = sentence_spans(text)
    ann = {k: {"choice": "NotMentioned", "spans": []} for k in NLI_IDS}
    for k, (choice, idx) in labels.items():
        ann[k] = {"choice": choice, "spans": idx}
    return {"id": doc_id, "file_name": f"nda-{doc_id}.pdf", "text": text, "spans": spans,
            "annotation_sets": [{"annotations": ann}], "document_type": "search-pdf", "url": "https://example.com"}


def main():
    squad = {"version": "aok_v1.0", "data": []}
    for title, text in (("AcmeWidgets_MSA", ACME), ("Northwind_Services", NORTHWIND)):
        qas = []
        for cat in ALL_CATEGORIES:
            answers = [{"text": q, "answer_start": text.index(q)} for q in CUAD_ANSWERS[title].get(cat, [])]
            qas.append({"answers": answers, "id": f"{title}__{cat}", "question": f'Highlight the parts (if any) of '
                        f'this contract related to "{cat}" that should be reviewed by a lawyer.',
                        "is_impossible": not answers})
        squad["data"].append({"title": title, "paragraphs": [{"qas": qas, "context": text}]})
    (HERE / "cuad_mini.json").write_text(json.dumps(squad, indent=1))
    test = {"version": "aok_v1.0", "data": [d for d in squad["data"] if d["title"] == "Northwind_Services"]}
    (HERE / "cuad_test_mini.json").write_text(json.dumps(test, indent=1))
    columns = ["Filename"] + sorted({k for row in MASTER for k in row} - {"Filename"})
    with open(HERE / "master_clauses_mini.csv", "w", newline="") as f:
        w = csv.DictWriter(f, columns)
        w.writeheader()
        w.writerows(MASTER)

    from contract_analyst.fields import NDA  # hypotheses as in the dataset's "labels"

    labels = {f.source: {"short_description": f.label, "hypothesis": f.description} for f in NDA if f.source}
    nli = {"documents": [
        nli_doc(1, NDA_1, {"nda-3": ("Entailment", [1]), "nda-4": ("Entailment", [2]), "nda-5": ("Entailment", [3]),
                           "nda-19": ("Entailment", [4])}),
        nli_doc(2, NDA_2, {"nda-1": ("Entailment", [1]), "nda-3": ("Contradiction", [1]),
                           "nda-11": ("Entailment", [2])}),
    ], "labels": labels}
    (HERE / "contractnli_mini.json").write_text(json.dumps(nli, indent=1))
    (HERE / "sample_contract.txt").write_text(ACME)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(HERE.parent.parent))
    main()
