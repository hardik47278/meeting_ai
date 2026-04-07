# Core
import os
import io
import tempfile
import datetime
import requests
import pandas as pd
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

# Chains + RAG (moved to langchain_classic in v1)
from langchain_classic.chains import RetrievalQA

# Text splitter (still in langchain_community or langchain_text_splitters)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loaders
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Embeddings + Vector store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# --------------------------
# LangChain Cache (in-memory, clears on restart)
# --------------------------
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

set_llm_cache(InMemoryCache())


# Optional: CrewAI (we'll fallback if it's not installed)
try:
    from crewai import Agent, Task, Crew
    from crewai.process import Process
    CREW_AVAILABLE = True
except Exception:
    CREW_AVAILABLE = False

# Optional: Todoist
try:
    from todoist_api_python.api import TodoistAPI
    TODOIST_AVAILABLE = True
except Exception:
    TODOIST_AVAILABLE = False

# --------------------------
# Constants / Globals
# --------------------------
MODEL_NAME = "gemma2-9b-it"  # <-- Only model we use
GEN_TEMP = 0.7               # generation / strategy / briefs
EXTRACT_TEMP = 0.3           # task extraction / focused outputs

# --------------------------
# Utilities (Telegram + Todoist helpers)
# --------------------------
def send_telegram_message(bot_token: str, chat_id: str, text: str):
    """Send a plain Markdown message via Telegram Bot API"""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        r = requests.post(url, json=payload, timeout=15)
        if r.ok:
            return {"ok": True}
        return {"ok": False, "error": r.text}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def todoist_add_tasks(api_key: str, tasks: list, project_name: str | None = None):
    """Add tasks to Todoist (if library available). Returns dict with results or error."""
    if not TODOIST_AVAILABLE:
        return {"ok": False, "error": "todoist_api_python is not installed."}
    try:
        api = TodoistAPI(api_key)
        project_id = None
        if project_name:
            # find or create project
            projects = api.get_projects()
            for p in projects:
                if p.name.strip().lower() == project_name.strip().lower():
                    project_id = p.id
                    break
            if not project_id:
                proj = api.add_project(name=project_name)
                project_id = proj.id
        created = []
        for t in tasks:
            if not t.strip():
                continue
            created_task = api.add_task(content=t.strip(), project_id=project_id)
            created.append({"id": created_task.id, "content": t.strip(), "project_id": project_id})
        return {"ok": True, "created": created}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# --------------------------
# Session State Init
# --------------------------
def initialize_session_state():
    defaults = {
        "groq_api_key": "",
        "todoist_api_key": "",
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "setup": None,  # dict with meeting config
        "prepared": False,
        "vectorstore": None,
        "context_analysis": None,
        "meeting_strategy": None,
        "executive_brief": None,
        "task_extraction_results": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# --------------------------
# Document Processing
# --------------------------
def process_documents(base_context: str, uploaded_files):
    docs = []
    # base context temp file
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp:
        tmp.write(base_context)
        tmp.flush()
        docs.extend(TextLoader(tmp.name).load())

    # uploaded files
    if uploaded_files:
        for f in uploaded_files:
            suffix = f.name.split(".")[-1].lower()
            # write buffer to temp file for loader compatibility
            with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=False) as tf:
                tf.write(f.getbuffer())
                tf.flush()
                try:
                    if suffix == "pdf":
                        loader = PyPDFLoader(tf.name)
                    else:
                        loader = TextLoader(tf.name)
                    docs.extend(loader.load())
                    st.success(f"Processed: {f.name}")
                except Exception as e:
                    st.error(f"Error processing {f.name}: {e}")
    return docs

def create_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(splits, embeddings)

# --------------------------
# LLM Helpers
# --------------------------
def build_llm(temperature: float = GEN_TEMP):
    return ChatGroq(model=MODEL_NAME, temperature=temperature, groq_api_key=st.session_state["groq_api_key"])

def fallback_analysis(setup: dict):
    """Generate three markdown docs (context, strategy, brief) with plain prompts"""
    attendees_text = "\n".join(f"- {a}" for a in setup.get("attendees", []))
    llm = build_llm(GEN_TEMP)

    context_prompt = f"""You are a senior business analyst.
Create a thorough context analysis for the upcoming meeting.

Company: {setup['company']}
Objective: {setup['objective']}
Date: {setup['date']}
Duration: {setup['duration']} minutes
Focus Areas: {setup['focus']}

Attendees:
{attendees_text}

OUTPUT FORMAT: Markdown with sections:
- Executive Summary
- Company Background
- Situation Analysis
- Key Stakeholders
- Strategic Considerations
- Risks & Open Questions
"""

    strategy_prompt = f"""You are an expert facilitator.
Create a time-boxed meeting strategy for a {setup['duration']} minute meeting with {setup['company']}.

Objective: {setup['objective']}
Focus Areas: {setup['focus']}

OUTPUT FORMAT: Markdown with sections:
- Meeting Overview
- Detailed Agenda (with start times and durations)
- Key Talking Points
- Discussion Questions
- Roles & Owners
- Success Criteria
"""

    brief_prompt = f"""You are an executive communications expert.
Create an executive brief that highlights what leadership needs to know.

Company: {setup['company']}
Objective: {setup['objective']}
Focus Areas: {setup['focus']}

OUTPUT FORMAT: Markdown with sections:
- Executive Summary
- Key Talking Points
- Recommendations
- Anticipated Questions & Suggested Answers
- Next Steps (with owners and target dates)
"""

    context_md = llm.invoke(context_prompt).content
    strategy_md = llm.invoke(strategy_prompt).content
    brief_md = llm.invoke(brief_prompt).content
    return context_md, strategy_md, brief_md

def run_crewai_analysis(setup: dict):
    """If CrewAI is available, produce three docs via coordinated agents; else fallback."""
    if not CREW_AVAILABLE:
        return fallback_analysis(setup)

    attendees_text = "\n".join(f"- {a}" for a in setup.get("attendees", []))
    llm = build_llm(GEN_TEMP)

    context_agent = Agent(
        role="Context Analyst",
        goal="Provide comprehensive context analysis for the meeting",
        backstory="Expert business analyst who prepares context documents and identifies key stakeholders.",
        llm=llm,
        verbose=True
    )
    strategy_agent = Agent(
        role="Meeting Strategist",
        goal="Create detailed meeting strategy and agenda",
        backstory="Seasoned meeting facilitator who structures effective discussions and allocates time optimally.",
        llm=llm,
        verbose=True
    )
    brief_agent = Agent(
        role="Executive Briefer",
        goal="Generate concise executive briefings with actionable insights",
        backstory="Master communicator crafting crisp, leadership-ready briefs.",
        llm=llm,
        verbose=True
    )

    context_task = Task(
        description=f"""Analyze the context for the meeting with {setup['company']}.
Consider:
1) Company background and market position
2) Meeting objective: {setup['objective']}
3) Attendees:
{attendees_text}
4) Focus areas: {setup['focus']}

FORMAT: Markdown. Sections:
- Executive Summary
- Company Background
- Situation Analysis
- Key Stakeholders
- Strategic Considerations
- Risks & Open Questions
""",
        agent=context_agent
    )

    strategy_task = Task(
        description=f"""Develop a meeting strategy for the {setup['duration']}-minute meeting with {setup['company']}.
Include:
1) Time-boxed agenda with allocations
2) Key talking points per section
3) Discussion questions and role assignments
FORMAT: Markdown with headings.""",
        agent=strategy_agent
    )

    brief_task = Task(
        description=f"""Create an executive briefing for the meeting with {setup['company']}.
Include:
1) Executive summary with key points
2) Key talking points and recommendations
3) Anticipated questions and prepared answers
FORMAT: Markdown with headings.""",
        agent=brief_agent
    )

    crew = Crew(
        agents=[context_agent, strategy_agent, brief_agent],
        tasks=[context_task, strategy_task, brief_task],
        verbose=True,
        process=Process.sequential
    )

    # Crew returns a combined result; we’ll try to split conservatively,
    # and if that fails, we’ll run the fallback.
    try:
        result = crew.kickoff()
        # result can be a string or list depending on crewai version
        if isinstance(result, list) and len(result) >= 3:
            return str(result[0]), str(result[1]), str(result[2])
        elif isinstance(result, str):
            # If single string, heuristic split by headings
            parts = [p.strip() for p in result.split("\n# ") if p.strip()]
            if len(parts) >= 3:
                return "# " + parts[0], "# " + parts[1], "# " + parts[2]
            # Too ambiguous -> fallback
            return fallback_analysis(setup)
        else:
            return fallback_analysis(setup)
    except Exception:
        return fallback_analysis(setup)

def create_qa_chain(vectorstore):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Use the following context to answer the question. "
            "If you don't know, say you don't know.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = build_llm(GEN_TEMP)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

def extract_tasks_from_text(text: str):
    """LLM-based task extraction with simple, robust constraints."""
    llm = build_llm(EXTRACT_TEMP)
    prompt = f"""Extract actionable tasks from the following content.

REQUIREMENTS:
- Return tasks as a numbered list, one task per line.
- Keep each task short and specific.
- If no tasks, return '0. No actionable tasks found.'

CONTENT:
{text}
"""
    raw = llm.invoke(prompt).content
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    # Normalize: only keep lines that look like numbered items, strip numbers
    tasks = []
    for l in lines:
        # Accept with or without numbering; we'll strip leading digits.
        cleaned = l
        # remove leading numbering like "1. " or "1) " or "- "
        for token in [") ", ". ", "- "]:
            if cleaned[:3].startswith(token) or cleaned[1:3] == token:
                pass  # handled below
        # generic strip of leading digits and punctuation
        while cleaned and (cleaned[0].isdigit() or cleaned[0] in [")", ".", "-", "•"]):
            cleaned = cleaned[1:].lstrip()
        if cleaned:
            tasks.append(cleaned)
    # Deduplicate while preserving order
    seen = set()
    unique_tasks = []
    for t in tasks:
        if t not in seen:
            seen.add(t)
            unique_tasks.append(t)
    # handle empty
    if not unique_tasks:
        return []
    # Also trim "No actionable tasks found." if present
    return [t for t in unique_tasks if "no actionable tasks" not in t.lower()]

# --------------------------
# Streamlit UI
# --------------------------
def main():
    st.set_page_config(page_title="AI Meeting Assistant", page_icon="📝", layout="wide")
    st.title("📝 AI Meeting Assistant (Groq • gemma2-9b-it)")

    initialize_session_state()

    # Sidebar: Keys & Integrations
    with st.sidebar:
        st.subheader("API Keys")
        groq_api_key = st.text_input("Groq API Key", type="password", value=st.session_state["groq_api_key"])
        if groq_api_key:
            st.session_state["groq_api_key"] = groq_api_key
            os.environ["GROQ_API_KEY"] = groq_api_key

        todoist_api_key = st.text_input("Todoist API Key (optional)", type="password", value=st.session_state["todoist_api_key"])
        if todoist_api_key != st.session_state["todoist_api_key"]:
            st.session_state["todoist_api_key"] = todoist_api_key

        with st.expander("Telegram (optional)"):
            telegram_bot_token = st.text_input("Telegram Bot Token", type="password", value=st.session_state["telegram_bot_token"])
            telegram_chat_id = st.text_input("Telegram Chat ID", value=st.session_state["telegram_chat_id"])
            if telegram_bot_token != st.session_state["telegram_bot_token"]:
                st.session_state["telegram_bot_token"] = telegram_bot_token
            if telegram_chat_id != st.session_state["telegram_chat_id"]:
                st.session_state["telegram_chat_id"] = telegram_chat_id

        st.caption("This app prepares for meetings by analyzing docs, generating agendas & briefs, answering questions, and extracting tasks. All LLM calls use gemma2-9b-it on Groq.")

    # Tabs
    tab_setup, tab_results, tab_qa, tab_tasks = st.tabs([
        "Meeting Setup", "Preparation Results", "Q&A Assistant", "Task Management"
    ])

    # --- Meeting Setup ---
    with tab_setup:
        st.subheader("Meeting Configuration")
        cols = st.columns(2)
        with cols[0]:
            company_name = st.text_input("Company Name")
            meeting_date = st.date_input("Meeting Date", value=datetime.date.today())
            meeting_duration = st.slider("Meeting Duration (minutes)", 15, 240, 60, step=5)
        with cols[1]:
            meeting_objective = st.text_area("Meeting Objective", height=100)
            focus_areas = st.text_area("Focus Areas or Concerns", height=100)

        st.subheader("Attendees")
        attendees_df = st.data_editor(
            pd.DataFrame({"Name": [""], "Role": [""], "Company": [""]}),
            num_rows="dynamic", use_container_width=True
        )

        st.subheader("Documents")
        uploaded_files = st.file_uploader("Upload Documents", type=["txt", "pdf"], accept_multiple_files=True)

        if st.button("Prepare Meeting", type="primary", use_container_width=True):
            if not st.session_state["groq_api_key"] or not company_name or not meeting_objective:
                st.error("Please provide Groq API key, Company Name, and Meeting Objective.")
            else:
                attendees_list = []
                for _, row in attendees_df.iterrows():
                    if str(row["Name"]).strip():
                        attendees_list.append(f"{row['Name']}, {row['Role']}, {row['Company']}")
                st.session_state["setup"] = {
                    "company": company_name,
                    "objective": meeting_objective,
                    "date": meeting_date,
                    "duration": meeting_duration,
                    "attendees": attendees_list,
                    "focus": focus_areas,
                    "files": uploaded_files
                }
                st.session_state["prepared"] = False
                st.rerun()

    # --- Preparation Results ---
    with tab_results:
        if st.session_state["setup"] and not st.session_state["prepared"]:
            with st.status("Processing meeting data...", expanded=True) as status:
                setup = st.session_state["setup"]
                attendees_text = "\n".join(f"- {a}" for a in setup["attendees"])
                base_context = f"""
Meeting Information:
- Company: {setup['company']}
- Objective: {setup['objective']}
- Date: {setup['date']}
- Duration: {setup['duration']} minutes
- Focus Areas: {setup['focus']}

Attendees:
{attendees_text}
""".strip()

                # Build docs + vectorstore
                docs = process_documents(base_context, setup["files"])
                vectorstore = create_vectorstore(docs)
                st.session_state["vectorstore"] = vectorstore

                # Generate three outputs (CrewAI if available else fallback)
                try:
                    context_md, strategy_md, brief_md = run_crewai_analysis(setup)
                except Exception as e:
                    st.warning(f"CrewAI path failed, using fallback. Error: {e}")
                    context_md, strategy_md, brief_md = fallback_analysis(setup)

                st.session_state["context_analysis"] = context_md
                st.session_state["meeting_strategy"] = strategy_md
                st.session_state["executive_brief"] = brief_md
                st.session_state["prepared"] = True
                status.update(label="Meeting preparation complete!", state="complete", expanded=False)

        if st.session_state["prepared"]:
            r1, r2, r3 = st.tabs(["Context Analysis", "Meeting Strategy", "Executive Brief"])
            with r1:
                if st.session_state["context_analysis"]:
                    st.markdown(st.session_state["context_analysis"])
                else:
                    st.warning("No context analysis.")
            with r2:
                if st.session_state["meeting_strategy"]:
                    st.markdown(st.session_state["meeting_strategy"])
                else:
                    st.warning("No meeting strategy.")
            with r3:
                if st.session_state["executive_brief"]:
                    st.markdown(st.session_state["executive_brief"])
                else:
                    st.warning("No executive brief.")

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.session_state["context_analysis"]:
                    st.download_button("Download Context Analysis",
                                       data=st.session_state["context_analysis"],
                                       file_name="context_analysis.md",
                                       use_container_width=True)
            with c2:
                if st.session_state["meeting_strategy"]:
                    st.download_button("Download Meeting Strategy",
                                       data=st.session_state["meeting_strategy"],
                                       file_name="meeting_strategy.md",
                                       use_container_width=True)
            with c3:
                if st.session_state["executive_brief"]:
                    st.download_button("Download Executive Brief",
                       
