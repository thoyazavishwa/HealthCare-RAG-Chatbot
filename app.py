import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA  # updated import path
from chromadb.config import Settings

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in environment variables")

VECTORSTORE_DIR = "vectorstore/chroma"
CHROMA_COLLECTION = "medical_documents"

# -------------------------------------------------
# Streamlit Config
# -------------------------------------------------
st.set_page_config(
    page_title="Clinical Guidelines RAG Assistant",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Clinical Guidelines RAG Assistant")
st.caption("Doctor • Pharmacist • Patient | Evidence-based PDF answers")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
mode = st.sidebar.selectbox(
    "Response Mode",
    ["Doctor", "Pharmacist", "Patient"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("📄 **Knowledge Sources**")
st.sidebar.markdown("- Clinical Guidelines PDFs")
st.sidebar.markdown("- Drug Interactions PDFs")

# -------------------------------------------------
# Efficient Chat Memory (Summary + Sliding Window)
# -------------------------------------------------

# Ensure memory state is initialized before sidebar access
if "chat_summary" not in st.session_state:
    st.session_state.chat_summary = ""
if "recent_messages" not in st.session_state:
    st.session_state.recent_messages = []

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []

SUMMARY_TRIGGER_TURNS = 5


#-------------------------------------------------
# Display Memory Status in Sidebar

st.sidebar.markdown("---")
st.sidebar.markdown("🧠 **Conversation Memory**")
st.sidebar.write(
    f"Summary stored: {'Yes' if st.session_state.chat_summary else 'No'}"
)
st.sidebar.write(
    f"Recent turns: {len(st.session_state.recent_messages)//2}"
)


# -------------------------------------------------
# Persona Prompts
# -------------------------------------------------
PROMPTS = {

"Doctor": """
You are an experienced, empathetic senior physician.

PRIMARY RULE (very important):
- Always answer the user’s question directly first and then provide additional context or explanation.

Your role:
- Explain clinical guidelines clearly and thoughtfully
- Speak like a human expert, not a textbook
- You may rephrase, interpret, and connect ideas
- Keep a professional but conversational tone

Medical guardrails:
- Do NOT diagnose individual patients
- Do NOT prescribe or recommend specific treatment plans
- Do NOT go beyond the provided guideline context
- If information is missing, say so explicitly
- If a question suggests an emergency, clearly advise urgent medical care

How to answer:
- Start with a short, natural explanation
- Then explain what the guideline says and why
- Mention risks, red flags, and limitations when relevant
- It’s okay to say “In general…” or “According to the guideline…”

Formatting rules:
- If the user explicitly asks for a table, comparison, or tabular format,
  respond ONLY with a markdown table.
- Do NOT add explanations outside the table.
- If a value is not present in the context, write "Not specified".
- If no table is requested, respond normally in text.

Context:
{context}
""",

"Pharmacist": """
You are a knowledgeable, patient-focused clinical pharmacist.

PRIMARY RULE (very important):
- Always answer the user’s question directly first and then provide additional context or explanation.

Your role:
- Explain medications, interactions, and safety clearly
- Sound approachable and practical, not robotic
- Translate guideline language into real-world understanding

Medical guardrails:
- Do NOT diagnose diseases
- Do NOT recommend starting or stopping medications
- Avoid patient-specific dosing advice
- Stick strictly to the provided guideline context
- If unsure, state uncertainty clearly

How to answer:
- Explain the reasoning behind safety concerns
- Use examples when helpful (without personalizing)
- Highlight monitoring, interactions, and precautions
- Maintain a calm, supportive tone

Formatting rules:
- If the user explicitly asks for a table, comparison, or tabular format,
  respond ONLY with a markdown table.
- Do NOT add explanations outside the table.
- If a value is not present in the context, write "Not specified".
- If no table is requested, respond normally in text.

Context:
{context}
""",

"Patient": """
You are a friendly, trustworthy patient educator.

PRIMARY RULE:
- Always answer the question directly in simple language and then provide additional context or explanation.

Your role:
- Explain medical information in simple, human language
- Be reassuring, respectful, and easy to understand
- You may use analogies or everyday explanations

Medical guardrails:
- Do NOT diagnose or suggest treatments
- Do NOT give medical instructions
- Do NOT replace a healthcare professional
- If symptoms sound serious, advise seeing a doctor urgently
- Only use information present in the provided context

How to answer:
- Avoid medical jargon (or explain it if unavoidable)
- Focus on “what this means” rather than “what to do”
- Encourage discussion with healthcare professionals

Formatting rules:
- If the user explicitly asks for a table, comparison, or tabular format,
  respond ONLY with a markdown table.
- Do NOT add explanations outside the table.
- If a value is not present in the context, write "Not specified".
- If no table is requested, respond normally in text.

Context:
{context}
"""
}


prompt = PromptTemplate(
    template=PROMPTS[mode],
    input_variables=["context"]
)

# -------------------------------------------------
# Load Vector Store (Windows-safe)
# -------------------------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=OPENAI_API_KEY
    )

    client = chromadb.PersistentClient(
        path=VECTORSTORE_DIR,
        settings=Settings(
            anonymized_telemetry=False
        )
    )

    vectordb = Chroma(
        client=client,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION
    )
    return vectordb

vectordb = load_vectorstore()

# -------------------------------------------------
# LLM
# -------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    openai_api_key=OPENAI_API_KEY
)

# -------------------------------------------------
# RetrievalQA Chain
# -------------------------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# -------------------------------------------------
# Chat Memory Functions
# -------------------------------------------------

def build_contextual_question(user_query):       # this adds both the recent messages and the summary to the prompt. when turns exceed threshold, the summary is updated and recent messages cleared.
    parts = []

    if st.session_state.chat_summary:
        parts.append(
            "Conversation summary so far:\n"
            + st.session_state.chat_summary
        )

    if st.session_state.recent_messages:
        recent_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in st.session_state.recent_messages
        )
        parts.append("Recent conversation:\n" + recent_text)

    parts.append("Current question:\n" + user_query)
    return "\n\n".join(parts)


def summarize_chat():
    convo = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in st.session_state.recent_messages
    )

    summary_prompt = f"""
    Summarize the following conversation briefly.
    Preserve important clinical context and user intent.
    Avoid unnecessary detail.

    Conversation:
    {convo}
    """

    summary_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    summary = summary_llm.invoke(summary_prompt).content

    st.session_state.chat_summary = summary.strip()  # Update summary
    st.session_state.recent_messages = []            # Clear recent messages


def should_retrieve(user_query):         # a small LLM (4o-mini) to do intent novelty detection, NOT retrieval
    """ 
    Decide if new retrieval is needed or
    if the answer exists in chat memory.
    """

    # If no memory at all, must retrieve
    if not st.session_state.chat_summary and not st.session_state.recent_messages:
        return True

    memory_text = ""

    if st.session_state.chat_summary:
        memory_text += "Summary:\n" + st.session_state.chat_summary + "\n\n"

    if st.session_state.recent_messages:
        memory_text += "Recent conversation:\n"
        memory_text += "\n".join(
            f"{m['role']}: {m['content']}"
            for m in st.session_state.recent_messages
        )

    decision_prompt = f"""
    You are deciding whether a medical assistant needs to look up documents again.

    Conversation memory:
    {memory_text}

    New user question:
    {user_query}

    Question:
    Is the answer already available in the conversation memory?

    Answer ONLY one word:
    YES or NO
    """

    judge_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    decision = judge_llm.invoke(decision_prompt).content.strip().upper()
    return decision == "NO"   # NO → need retrieval



# -------------------------------------------------
# Render previous chat messages (UI only) as Streamlit reruns the entire script from top to bottom every time user interacts with it
# -------------------------------------------------
for msg in st.session_state.ui_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------------------------
# Chat Input
# -------------------------------------------------
user_query = st.chat_input("Ask a question from clinical guidelines...")

if user_query:

    # Emergency detection
    EMERGENCY_KEYWORDS = [
        "chest pain", "shortness of breath", "unconscious",
        "severe bleeding", "collapse"
    ]

    if any(k in user_query.lower() for k in EMERGENCY_KEYWORDS):
        st.warning(
            "⚠️ This may be a medical emergency. Please seek immediate medical care."
        )

    # Start: Show user message in chat UI
    with st.chat_message("user"):
        st.markdown(user_query)

    # Add user message to recent memory
    st.session_state.recent_messages.append(
        {"role": "user", "content": user_query}
    )

    st.session_state.ui_messages.append(
        {"role": "user", "content": user_query}
    )

    needs_retrieval = should_retrieve(user_query)

    with st.chat_message("assistant"):
        if needs_retrieval:
            with st.spinner("Consulting Healthcare guidelines..."):
                contextual_question = build_contextual_question(user_query)
                result = qa_chain(contextual_question)
                answer = result["result"]

                st.markdown(answer)

                # Save assistant response
                st.session_state.recent_messages.append(
                    {"role": "assistant", "content": answer}
                )
                st.session_state.ui_messages.append(
                    {"role": "assistant", "content": answer}
                )

                # Show sources ONLY when retrieval happened
                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(result["source_documents"], start=1):
                        st.markdown(
                            f"**Source {i}:** {doc.metadata.get('source', 'PDF')} "
                            f"(Page {doc.metadata.get('page', 'N/A')})"
                        )
                        st.write(doc.page_content[:500] + "...")

        else:
            # Answer from memory only (no RAG)
            memory_answer_prompt = f"""
            Use the conversation memory to answer naturally.

            Conversation summary:
            {st.session_state.chat_summary}

            Recent conversation:
            {st.session_state.recent_messages}

            User question:
            {user_query}
            """

            answer = llm.invoke(memory_answer_prompt).content

            st.markdown(answer)

            st.session_state.recent_messages.append(
                {"role": "assistant", "content": answer}
            )
            st.session_state.ui_messages.append(
                {"role": "assistant", "content": answer}
            )

            st.caption("ℹ️ Answered from conversation memory (no document lookup)")


    # Summarize occasionally
    if len(st.session_state.recent_messages) >= SUMMARY_TRIGGER_TURNS * 2:
        summarize_chat()


# -------------------------------------------------
# Footer Safety Notice
# -------------------------------------------------
st.markdown("---")
st.caption(
    "⚠️ This assistant provides guideline-based information only. "
    "It does not replace professional medical judgment."
)
