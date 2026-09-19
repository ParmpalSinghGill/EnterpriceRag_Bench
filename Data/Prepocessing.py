import numpy as np
from datasets import load_dataset

SAMPLES_PER_TYPE = 5
CONTENT_PREVIEW = 1000

ds = load_dataset("onyx-dot-app/EnterpriseRAG-Bench", "documents")
documents = ds["test"]

ds = load_dataset("onyx-dot-app/EnterpriseRAG-Bench", "questions")
questions = ds["test"]


def print_source_type_counts(documents):
    types, counts = np.unique(documents["source_type"], return_counts=True)
    print("Source types:")
    for source_type, count in zip(types, counts):
        print(f"  {source_type}: {count}")
    return types


def print_document_samples(documents, source_types, samples_per_type=SAMPLES_PER_TYPE):
    for source_type in source_types:
        idxs = [i for i, x in enumerate(documents["source_type"]) if x == source_type]
        print("\n" + "=" * 80)
        print(f"DOCUMENTS — SOURCE TYPE: {source_type}  (n={len(idxs)})")
        print("=" * 80)

        for i, idx in enumerate(idxs[:samples_per_type]):
            row = documents[idx]
            content = row["content"]
            if len(content) > CONTENT_PREVIEW:
                content = content[:CONTENT_PREVIEW] + "..."

            print(f"\n--- sample {i + 1} (idx={idx}) ---")
            print(f"doc_id: {row['doc_id']}")
            print(f"title: {row['title']}")
            print(f"content: {content}")


def print_question_samples(questions, source_types, samples_per_type=SAMPLES_PER_TYPE):
    for source_type in source_types:
        idxs = [
            i
            for i, types in enumerate(questions["source_types"])
            if source_type in types
        ]
        print("\n" + "=" * 80)
        print(f"QUESTIONS — SOURCE TYPE: {source_type}  (n={len(idxs)})")
        print("=" * 80)

        np.random.shuffle(idxs)
        for i, idx in enumerate(idxs[:samples_per_type]):
            row = questions[idx]
            print(f"\n--- sample {i + 1} (idx={idx}) ---")
            print(f"question_id: {row['question_id']}")
            print(f"question_type: {row['question_type']}")
            print(f"source_types: {row['source_types']}")
            print(f"question: {row['question']}")
            print(f"expected_doc_ids: {row['expected_doc_ids']}")
            print(f"gold_answer: {row['gold_answer']}")
            print(f"answer_facts: {row['answer_facts']}")


if __name__ == "__main__":
    types=print_source_type_counts(documents)

    # types = ["jira"]
    # print_document_samples(documents, types)
    print_question_samples(questions, types)
