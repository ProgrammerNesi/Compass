import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from retrieval.retrieval import retrieve


load_dotenv(override=True)


MODEL = "gemini-3.1-flash-lite"


llm = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0
)


SYSTEM_PROMPT = """
You are a helpful document assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not invent information.
- If the context does not contain enough information, say so.
- Give a direct and concise answer.
- Cite sources when possible.

Context:

{context}
"""


def answer_question(question):

    docs = retrieve(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT.format(
                context=context
            )
        ),
        HumanMessage(
            content=question
        )
    ]

    response = llm.invoke(messages)

    return response.content, docs
