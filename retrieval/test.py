from answer import answer_question


question = (
    "Who founded Insurellm, when was it founded, "
    "and what was its first product?"
)


answer, docs = answer_question(question)


print("\nANSWER")
print("=" * 80)
print(answer)


print("\nRETRIEVED CONTEXT")
print("=" * 80)

for i, doc in enumerate(docs, 1):

    print(f"\n--- Document {i} ---")

    print(
        doc.metadata.get("source")
    )

    print(
        doc.page_content[:500]
    )