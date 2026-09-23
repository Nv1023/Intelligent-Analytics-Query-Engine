import json
from pathlib import Path

from analytics_engine import QueryEngine


def run_sample_queries(engine):
    data_dir = Path(__file__).resolve().parent / "dataset"
    queries = json.loads((data_dir / "nl_queries.json").read_text(encoding="utf-8"))
    results = engine.answer_queries(queries)
    for item in results:
        print(json.dumps(item, indent=2, default=str))
        print("-" * 80)


def run_manual_chat(engine):
    print("\nSQL Analytics Chatbot")
    print("Type 'exit' to quit.")
    while True:
        question = input("\nAsk a sales question: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if not question:
            continue
        answer = engine.chat_answer(question)
        print("\nGenerated SQL:")
        print(answer["generated_sql"])
        print("\nResult:")
        print(json.dumps(answer["result"], indent=2, default=str))
        print("\nConfidence:", answer["confidence_score"])
        print("\nExplanation:", answer["explanation"])


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "dataset"
    engine = QueryEngine(str(data_dir))

    mode = input("Choose mode: [1] sample queries [2] manual chat: ").strip()
    if mode == "2":
        run_manual_chat(engine)
    else:
        run_sample_queries(engine)
