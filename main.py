from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import json, re, sys, os
from openai import OpenAI
from docx import Document

def transcribe(path):
    client = OpenAI()   # uses OPENAI_API_KEY from .env
    with open(path, "rb") as f:
        return client.audio.transcriptions.create(model="whisper-1", file=f).text

load_dotenv()


class KeyFigures(BaseModel):
    totalInvestment: Optional[str] = None
    portfolioValue: Optional[str] = None
    portfolioPct: Optional[str] = None
    platformName: Optional[str] = None
    platformValue: Optional[str] = None
    incomeValue: Optional[str] = None
    incomePct: Optional[str] = None


class AllocationItem(BaseModel):
    class_: Optional[str] = Field(default=None, alias="class")
    key: Optional[str] = None
    percent: Optional[float] = None


class Strategy(BaseModel):
    model: Optional[str] = None
    portfolioName: Optional[str] = None
    currency: Optional[str] = None
    allocation: Optional[List[AllocationItem]] = None
    composition: Optional[str] = None
    rationale: Optional[str] = None
    keyFigures: Optional[KeyFigures] = None


class GoalAssessment(BaseModel):
    clientType: Optional[str] = None
    mandateType: Optional[str] = None
    wrapper: Optional[str] = None
    currency: Optional[str] = None
    portfolioModel: Optional[str] = None
    investmentStrategy: Optional[str] = None
    rationale: Optional[str] = None
    strategyBullets: Optional[str] = None
    keyFigures: Optional[KeyFigures] = None


class Objective(BaseModel):
    targetReturn: Optional[str] = None
    benchmark: Optional[str] = None
    riskProfile: Optional[str] = None
    strategyChars: Optional[str] = None
    investmentHorizon: Optional[str] = None


class Fees(BaseModel):
    advisorFee: Optional[str] = None
    management: Optional[str] = None
    adminFee: Optional[str] = None
    brokerage: Optional[str] = None


class Replacements(BaseModel):
    isReplacement: Optional[bool] = None
    details: Optional[str] = None


class ProposalResponse(BaseModel):
    clientName: Optional[str] = None
    clientEmail: Optional[str] = None
    introGreetTo: Optional[Literal["client", "advisor"]] = None
    applyModel: Optional[Literal["balanced", "global-growth", "income"]] = None
    introduction: Optional[str] = None
    needs: Optional[str] = None
    goalAssessment: Optional[GoalAssessment] = None
    strategy: Optional[Strategy] = None
    objective: Optional[Objective] = None
    fees: Optional[Fees] = None
    replacements: Optional[Replacements] = None
    status: Optional[Literal["draft", "complete"]] = None

    model_inferred: bool = Field(
        default=False,
        description="True if the AI suggested a model rather than extracting it from the notes."
    )

    evidence: Dict[str, str] = Field(
        default_factory=dict,
        description="Dotted field path mapped to a supporting quote from the notes."
    )

RISKS = ["Conservative", "Moderately Conservative", "Moderate",
         "Moderately Aggressive", "Aggressive"]


def amount(s):
    if not s:
        return None

    t = str(s).lower().replace(",", "").replace(" ", "")

    m = re.search(r"(\d+(?:\.\d+)?)(million|m|k)?", t)

    if not m:
        return None

    multiplier = {
        "million": 1_000_000,
        "m": 1_000_000,
        "k": 1_000
    }.get(m.group(2), 1)

    value = float(m.group(1)) * multiplier

    return str(int(value))


def pct(s):
    if not s:
        return None

    m = re.search(r"\d+(?:\.\d+)?", str(s))

    return m.group(0) if m else None


def horizon(s):
    if not s:
        return None

    m = re.search(r"\d+", str(s))

    if not m:
        return None

    years = int(m.group(0))

    if years < 3:
        return "1 year+"
    elif years < 5:
        return "3 years+"
    else:
        return "5 years+"

def prune(d):
    if isinstance(d, dict):
        d = {k: prune(v) for k, v in d.items()}
        return {k: v for k, v in d.items() if v not in (None, {}, "")}
    return d

def get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d: return None
        d = d[k]
    return d


REQUIRED = {
    "clientName": "Client name",
    "strategy.keyFigures.totalInvestment": "Amount to invest",
    "strategy.keyFigures.incomeValue": "Monthly income need",
    "objective.investmentHorizon": "Time horizon",
    "objective.riskProfile": "Risk profile",
    "goalAssessment.currency": "Currency",
}


def build_output(p: ProposalResponse):

    # Convert the Pydantic model into a normal dictionary.
    # by_alias=True ensures class_ becomes "class".
    d = p.model_dump(by_alias=True)

    evidence = d.pop("evidence", {})
    inferred = d.pop("model_inferred", False)

    # Clean and standardise key financial figures.
    def clean_key_figures(key_figures):

        if not key_figures:
            return

        for field in ["totalInvestment", "portfolioValue",
                      "platformValue", "incomeValue"]:

            if key_figures.get(field):
                key_figures[field] = amount(key_figures[field])

        for field in ["portfolioPct", "incomePct"]:

            if key_figures.get(field):
                key_figures[field] = pct(key_figures[field])

    # Key figures may be stored under strategy or goalAssessment.
    strategy = d.get("strategy") or {}
    goal = d.get("goalAssessment") or {}

    clean_key_figures(strategy.get("keyFigures"))
    clean_key_figures(goal.get("keyFigures"))

    # Standardise fee percentages.
    fees = d.get("fees") or {}

    for field in ["advisorFee", "management", "adminFee", "brokerage"]:
        if fees.get(field):
            fees[field] = pct(fees[field])

    # Standardise investment horizon.
    objective = d.get("objective") or {}

    if objective.get("investmentHorizon"):
        objective["investmentHorizon"] = horizon(
            objective["investmentHorizon"]
        )

    # Check which required fields are missing.
    proposal = prune(d)

    missing = [
        label
        for path, label in REQUIRED.items()
        if get(proposal, path) is None
    ]

    # Mark the proposal as a draft until required information is present
    # and any AI-suggested model has been confirmed.
    proposal["status"] = "draft" if missing or inferred else "complete"

    return {
        "status": "draft" if missing or inferred else "ready_for_review",
        "proposal": proposal,
        "missing": missing,
        "model_suggested_by_ai": inferred,
        "evidence": evidence
    }

llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0, timeout=60, max_retries=1)
parser = PydanticOutputParser(pydantic_object=ProposalResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an AI assistant working for Meridian Private Clients,
            a fictional wealth management firm.

            Your task is to convert unstructured Portfolio Manager
            meeting notes into structured investment proposal data.

            EXTRACTION INSTRUCTIONS:

            1. Extract the client's full name and email address
               if provided.

            2. Extract relevant client information and financial needs.

            3. Identify the client's investment amount and currency.

            4. Extract the client's investment objectives.

            5. Extract the investment horizon, target return,
               benchmark and risk profile if mentioned.

            6. Extract the investment strategy and rationale
               discussed by the Portfolio Manager.

            7. Extract the applicable fees if provided.

            8. Identify the client type, mandate type and
               investment wrapper if mentioned.

            INVESTMENT MODEL OPTIONS:

            - balanced
            - global-growth
            - income

            If the PM did not name a model, you may suggest one only with
            clear support in the notes, and you MUST set model_inferred=true.
            riskProfileCategory must be one of: Conservative,
            Moderately Conservative, Moderate, Moderately Aggressive,
            Aggressive, or null.

            IMPORTANT RULES:

            - Never invent client information.
            - Never invent financial figures.
            - Never assume financial needs or investment objectives.
            - If the amount, horizon, income need or risk profile is not
              in the notes, return null. Do not guess.
            - For every non-null field, add its dotted path (e.g.
              strategy.keyFigures.totalInvestment) and a short verbatim
              quote from the notes to evidence.
            - Preserve the original meaning of the meeting notes.
            - If information is unavailable, return null.
            - Do not confuse an investment amount with portfolio value.
            - Do not calculate fees or returns unless explicitly
              instructed and sufficient information is provided.

            OUTPUT REQUIREMENTS:

            - Follow the supplied Pydantic output structure exactly.
            - Use the correct nested fields.
            - Do not include additional fields.
            - Return valid structured data.

            {format_instructions}
            """
        ),
        (
            "human",
            "Convert the following Portfolio Manager meeting notes "
            "into a structured investment proposal:\n\n{query}"
        )
    ]
).partial(
    format_instructions=parser.get_format_instructions()
)

chain = prompt | llm | parser

def build_docx(out, path="proposal.docx"):
    p = out["proposal"]
    doc = Document()
    doc.add_heading("Investment Proposal", 0)
    doc.add_paragraph("DRAFT for Portfolio Manager review")
    doc.add_paragraph("Client: " + p.get("clientName", "[to be provided]"))

    if p.get("introduction"):
        doc.add_heading("Introduction", 1)
        doc.add_paragraph(p["introduction"])
    if p.get("needs"):
        doc.add_heading("Client needs", 1)
        doc.add_paragraph(p["needs"])

    doc.add_heading("Key details", 1)
    rows = [
        ("Amount to invest", get(p, "strategy.keyFigures.totalInvestment")),
        ("Monthly income", get(p, "strategy.keyFigures.incomeValue")),
        ("Currency", get(p, "goalAssessment.currency")),
        ("Risk profile", get(p, "objective.riskProfileCategory")),
        ("Time horizon", get(p, "objective.investmentHorizon")),
        ("Target return", get(p, "objective.targetReturn")),
        ("Adviser fee (%)", get(p, "fees.advisorFee")),
        ("Model portfolio", p.get("applyModel")),
    ]
    table = doc.add_table(rows=0, cols=2)
    table.style = "Table Grid"
    for label, value in rows:
        cells = table.add_row().cells
        cells[0].text = label
        cells[1].text = str(value) if value else "[to be provided by PM]"

    doc.add_heading("Items for PM review", 1)
    for m in out["missing"]:
        doc.add_paragraph("Missing: " + m, style="List Bullet")
    if out["model_suggested_by_ai"]:
        doc.add_paragraph("Model portfolio was suggested by the AI: please confirm.", style="List Bullet")
    for field, quote in out["evidence"].items():
        doc.add_paragraph(f'{field}: "{quote}"', style="List Bullet")

    doc.save(path)
    return path

print("\n=== Meridian Private Clients | AI Proposal Builder ===")

print("\nPaste your Portfolio Manager meeting notes below.")
print("When finished, type END on a new line and press Enter.\n")

notes_lines = []

while True:
    line = input()

    if line.strip().upper() == "END":
        break

    notes_lines.append(line)

meeting_notes = "\n".join(notes_lines)

if not meeting_notes.strip():
    raise ValueError("No meeting notes were provided.")

try:
    result = chain.invoke({"query": meeting_notes})
    out = build_output(result)

    # --- JSON output ---
    with open("proposal.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(json.dumps(out, indent=2, ensure_ascii=False))

    if out["missing"]:
        print("\n=== PM must provide ===")
        for m in out["missing"]:
            print("-", m)

    if out["model_suggested_by_ai"]:
        print("\n- Model portfolio was suggested by the AI: please confirm.")

    # --- Load generated proposal into HTML tool ---
    print("\nWrote proposal.json")

    proposal_data = out["proposal"]

    with open("proposal_for_html.json", "w", encoding="utf-8") as f:
        json.dump(proposal_data, f, indent=2, ensure_ascii=False)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "challenge-generator.html")

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    # Set the browser tab and document title to the client's name
    client_name = proposal_data.get("clientName") or "New Client"
    html_content = re.sub(
    r"<title>.*?</title>",
    f"<title>{client_name} - Investment Proposal</title>",
    html_content,
    count=1,
    flags=re.IGNORECASE | re.DOTALL
)

    proposal_json = json.dumps(
        proposal_data,
        ensure_ascii=False
    ).replace("</", "<\\/")

    loader_script = f"""
<script>
window.addEventListener("load", function () {{
    try {{
        const proposal = {proposal_json};
        window.loadProposal(proposal);
        console.log("AI proposal successfully loaded.");
    }} catch (error) {{
        console.error("Could not load AI proposal:", error);
    }}
}});
</script>
"""

    if "</body>" not in html_content.lower():
        raise ValueError("Could not find closing body tag in HTML file.")

    html_content = re.sub(
        r"</body\s*>",
        loader_script + "\n</body>",
        html_content,
        count=1,
        flags=re.IGNORECASE
    )

    generated_html_path = os.path.join(
        base_dir,
        "generated_proposal.html"
    )

    with open(generated_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("Wrote proposal_for_html.json")
    print("Wrote generated_proposal.html")

    os.startfile(generated_html_path)

except Exception as e:                                                                           
    print("\nError generating proposal:", e)