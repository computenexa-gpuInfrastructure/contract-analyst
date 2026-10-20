"""The fields we extract: ONE source of truth for prompts, dataset mapping, scoring and reports.

Types:
  text         free text (governing law; parties = a list of names)
  date         a calendar date, answered as YYYY-MM-DD (or short text such as "perpetual")
  duration     a period such as "1 year" or "30 days"
  yesno        yes / no (a "yes" still needs a supporting quote)
  yesno_quote  yes / no where the quote is the main output (risk flags)
  nli          yes / no / not_mentioned for an NDA statement (ContractNLI hypothesis)

`source` links a field to its dataset: a CUAD category name or a ContractNLI label id ("nda-4").
`hint` = how to read the contract for that field, sent only with --instructions guided (NDA_HINTS).
Fields with source None are extracted but not scored yet ("synthetic data later").
Version 1 uses only CUAD categories with at least 50 examples.
"""
from dataclasses import dataclass

VERSION = 1
TYPES = ("text", "date", "duration", "yesno", "yesno_quote", "nli")
DOC_TYPES = ("commercial", "nda")


@dataclass(frozen=True)
class Field:
    id: str
    label: str
    type: str
    description: str  # goes into the prompt
    doc_type: str  # commercial | nda
    source: str | None = None  # CUAD category / ContractNLI label; None = not evaluated yet
    value_column: str | None = None  # CUAD master_clauses.csv column with the normalised gold value
    note: str = ""
    hint: str = ""  # reading rule, sent only with --instructions guided

    @property
    def evaluated(self) -> bool:
        return self.source is not None


def _c(id, label, type, description, source, value_column=None):
    return Field(id, label, type, description, "commercial", source, value_column)


# Reading rules for --instructions guided: ContractNLI's labelling conventions in plain words. Checked against the
# TRAIN split's labels; never tuned on the test split.
NDA_HINTS = {
    "limited_use": "Yes if the information may be used only for the purpose, transaction or project of the "
                   "agreement.",
    "share_with_employees": 'Yes if employees (or officers, directors, staff, representatives) may receive it, even '
                            'only on a need-to-know basis or under conditions: "shall not disclose except to its '
                            'employees who need to know" means yes.',
    "share_with_third_parties": "Third parties include consultants, contractors, agents, professional advisers, "
                                "lawyers, accountants and affiliates. Yes if the receiver may share with any of them "
                                "(even only on a need-to-know basis or if they are bound by confidentiality): "
                                '"shall not disclose to third parties except its advisers" means yes. No if every '
                                "disclosure outside the receiver needs the discloser's consent.",
    "notice_on_compelled_disclosure": "Yes only if the receiver must notify or inform the discloser when law, a "
                                      "court or a regulator requires disclosure. A mere permission to disclose when "
                                      "required by law, with no duty to notify, is not_mentioned.",
    "verbal_info_included": "Yes if the definition covers oral / verbal information, or information disclosed in "
                            "any form or however disclosed. No if only written or tangible information is covered.",
    "independent_development_allowed": "Usually in the list of information that is NOT confidential: yes if "
                                       "information the receiver develops independently is excluded or allowed.",
    "third_party_acquisition_allowed": "Usually in the list of information that is NOT confidential: yes if "
                                       "information the receiver gets from a third party or another source (without "
                                       "a duty of confidentiality, or without breach) is excluded or allowed.",
    "no_licence": "Yes if no licence or right in the information is granted, or the information stays the "
                  "discloser's property.",
    "survival_of_obligations": "Yes only if some obligations continue after the agreement ends or terminates (or "
                               'for a period after termination or disclosure). A plain term ("this agreement lasts '
                               '3 years") or an expiry date is not survival.',
    "return_or_destroy": "Yes only if return or destruction is required when the agreement ends, expires or "
                         "terminates (or when the dealings or purpose end). A return only on the discloser's "
                         "request, not linked to the end of the agreement, is not_mentioned.",
    "keep_copies_after_return": "Yes if the receiver may keep some information after returning or destroying it "
                                "(one archival copy, back-ups, copies required by law or for records). No only if it "
                                "expressly says no copies may be kept or retained. A duty to return all copies "
                                "without such words is not_mentioned.",
    "may_copy": 'Yes if copying is allowed in some cases, including "shall not copy except as needed for the '
                'purpose". No if copying is forbidden, or allowed only with the discloser\'s prior consent.',
    "must_be_marked": "Yes if only information marked or identified as confidential is protected (oral information "
                      "confirmed in writing counts as identified). No if information is protected without marking: "
                      "whether or not marked, however disclosed, or what a reasonable person would see as "
                      "confidential.",
    "nda_is_confidential": "Yes if the receiver must keep secret the existence or terms of this agreement, or the "
                           "fact that discussions or negotiations are taking place.",
    "non_solicit": "Yes if the receiver may not solicit, hire or entice the discloser's employees (or other "
                   "representatives).",
    "no_reverse_engineering": "Yes if reverse engineering, decompiling or disassembling is forbidden.",
    "technical_info_only": "Decide from the definition of Confidential Information: yes if it covers only technical "
                           "information; no if it also covers business, financial, commercial, customer or any other "
                           "information; not_mentioned only if there is no definition.",
}


def _n(id, label, source, hypothesis):
    return Field(id, label, "nli", hypothesis, "nda", source, hint=NDA_HINTS.get(id, ""))


COMMERCIAL = [
    _c("parties", "Parties", "text", "The parties who signed the contract (company or person names, not roles).",
       "Parties", "Parties-Answer"),
    _c("agreement_date", "Agreement date", "date", "The date the contract was made or signed.",
       "Agreement Date", "Agreement Date-Answer"),
    _c("effective_date", "Effective date", "date", "The date the contract starts to apply.",
       "Effective Date", "Effective Date-Answer"),
    _c("expiration_date", "Expiration date", "date", "The date the initial term ends (or 'perpetual').",
       "Expiration Date", "Expiration Date-Answer"),
    _c("renewal_term", "Renewal term", "duration", "How long each renewal lasts after the initial term "
       "(e.g. '1 year'); null if the contract does not renew.", "Renewal Term", "Renewal Term-Answer"),
    _c("notice_to_terminate_renewal", "Notice to stop renewal", "duration", "The notice needed to stop a renewal "
       "(e.g. '60 days').", "Notice Period To Terminate Renewal", "Notice Period To Terminate Renewal- Answer"),
    _c("governing_law", "Governing law", "text", "Which state's or country's law governs the contract "
       "(e.g. 'New York').", "Governing Law", "Governing Law-Answer"),
    _c("termination_for_convenience", "Termination for convenience", "yesno", "Can a party end the contract "
       "without cause (only by giving notice)?", "Termination For Convenience"),
    _c("cap_on_liability", "Cap on liability", "yesno_quote", "Is liability limited to an amount, a multiple of "
       "fees or a time period?", "Cap On Liability"),
    # Risk flags.
    _c("uncapped_liability", "Uncapped liability", "yesno_quote", "Is some liability expressly NOT limited, "
       "e.g. excluded from the cap (such as for gross negligence or confidentiality breaches)?",
       "Uncapped Liability"),
    _c("anti_assignment", "Anti-assignment", "yesno_quote", "Is consent or notice needed to assign the contract "
       "to someone else?", "Anti-Assignment"),
    _c("audit_rights", "Audit rights", "yesno_quote", "Can a party audit the other's books, records or premises?",
       "Audit Rights"),
    _c("exclusivity", "Exclusivity", "yesno_quote", "Is there an exclusive dealing commitment (e.g. buy only from, "
       "or sell only to, one party)?", "Exclusivity"),
    _c("insurance", "Insurance", "yesno_quote", "Must a party maintain insurance?", "Insurance"),
    _c("minimum_commitment", "Minimum commitment", "yesno_quote", "Is there a minimum order, purchase or "
       "payment commitment?", "Minimum Commitment"),
    _c("ip_ownership_assignment", "IP ownership assignment", "yesno_quote", "Does intellectual property created "
       "by one party become owned by the other?", "Ip Ownership Assignment"),
    _c("change_of_control", "Change of control", "yesno_quote", "Does a merger, acquisition or change of control "
       "give a right to terminate, or need consent?", "Change Of Control"),
    _c("non_compete", "Non-compete", "yesno_quote", "Is a party restricted from competing (in a market, area "
       "or time)?", "Non-Compete"),
    _c("liquidated_damages", "Liquidated damages", "yesno_quote", "Is there a fixed amount payable on breach or "
       "termination (liquidated damages or a termination fee)?", "Liquidated Damages"),
]

# The 17 ContractNLI hypotheses (labels nda-1 .. nda-20; 6, 9 and 14 are not in the dataset).
NDA = [
    _n("limited_use", "Limited use", "nda-4", "Receiving Party shall not use any Confidential Information for "
       "any purpose other than the purposes stated in Agreement."),
    _n("share_with_employees", "Share with employees", "nda-5", "Receiving Party may share some Confidential "
       "Information with some of Receiving Party's employees."),
    _n("share_with_third_parties", "Share with third parties", "nda-7", "Receiving Party may share some "
       "Confidential Information with some third-parties (including consultants, agents and professional "
       "advisors)."),
    _n("notice_on_compelled_disclosure", "Notice on compelled disclosure", "nda-8", "Receiving Party shall notify "
       "Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose "
       "any Confidential Information."),
    _n("verbal_info_included", "Verbal information included", "nda-3", "Confidential Information may include "
       "verbally conveyed information."),
    _n("independent_development_allowed", "Independent development allowed", "nda-12", "Receiving Party may "
       "independently develop information similar to Confidential Information."),
    _n("third_party_acquisition_allowed", "Third-party acquisition allowed", "nda-13", "Receiving Party may "
       "acquire information similar to Confidential Information from a third party."),
    _n("no_licence", "No licence", "nda-15", "Agreement shall not grant Receiving Party any right to "
       "Confidential Information."),
    _n("survival_of_obligations", "Obligations survive", "nda-19", "Some obligations of Agreement may survive "
       "termination of Agreement."),
    _n("return_or_destroy", "Return or destroy", "nda-16", "Receiving Party shall destroy or return some "
       "Confidential Information upon the termination of Agreement."),
    _n("keep_copies_after_return", "May keep copies after return", "nda-20", "Receiving Party may retain some "
       "Confidential Information even after the return or destruction of Confidential Information."),
    _n("may_copy", "May make copies", "nda-17", "Receiving Party may create a copy of some Confidential "
       "Information in some circumstances."),
    _n("must_be_marked", "Must be marked", "nda-1", "All Confidential Information shall be expressly identified "
       "by the Disclosing Party."),
    _n("nda_is_confidential", "The NDA itself is confidential", "nda-10", "Receiving Party shall not disclose the "
       "fact that Agreement was agreed or negotiated."),
    _n("non_solicit", "Non-solicit", "nda-18", "Receiving Party shall not solicit some of Disclosing Party's "
       "representatives."),
    _n("no_reverse_engineering", "No reverse engineering", "nda-11", "Receiving Party shall not reverse engineer "
       "any objects which embody Disclosing Party's Confidential Information."),
    _n("technical_info_only", "Technical information only", "nda-2", "Confidential Information shall only "
       "include technical information."),
    # Not in ContractNLI: extracted, scored later with synthetic data (no generator yet).
    Field("mutual_or_one_way", "Mutual or one-way", "text", "Is the NDA 'mutual' (both parties disclose) or "
          "'one-way' (only one party discloses)?", "nda", note="synthetic data later"),
    Field("term", "Term", "duration", "How long the NDA (or its confidentiality obligations) lasts, "
          "e.g. '2 years'.", "nda", note="synthetic data later"),
    Field("parties", "Parties", "text", "The parties who signed the NDA (names, not roles).", "nda",
          note="synthetic data later"),
    Field("governing_law", "Governing law", "text", "Which state's or country's law governs the NDA.", "nda",
          note="synthetic data later"),
]

FIELDS = {"commercial": COMMERCIAL, "nda": NDA}
DATASET_DOC_TYPE = {"cuad": "commercial", "contractnli": "nda"}

# ContractNLI label names -> our answers.
NLI_LABELS = {"Entailment": "yes", "Contradiction": "no", "NotMentioned": "not_mentioned"}


def fields_for(doc_type: str) -> list[Field]:
    if doc_type not in FIELDS:
        raise ValueError(f"unknown contract type {doc_type!r} (use one of {', '.join(DOC_TYPES)})")
    return FIELDS[doc_type]


def not_found_value(field: Field):
    """What a field's answer is when the contract says nothing about it."""
    return {"yesno": "no", "yesno_quote": "no", "nli": "not_mentioned"}.get(field.type)


def is_found(field: Field, value) -> bool:
    """A field is "found" when the answer is positive: a value, a "yes", or (for NDAs) yes / no."""
    if field.type in ("yesno", "yesno_quote"):
        return value == "yes"
    if field.type == "nli":
        return value in ("yes", "no")
    return value not in (None, "", [])
