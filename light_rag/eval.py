import os

import numpy as np
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from trulens.apps.custom import TruCustomApp, instrument
from trulens.core import Feedback, Select, TruSession
from trulens.providers.openai import OpenAI

load_dotenv()

session = TruSession()
session.reset_database()

provider = OpenAI(model_engine="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))


class InstrumentedLightRAG(LightRAG):
    @instrument
    def retrieve(self, text):
        return super().retrieve(text)

    @instrument
    def query(self, text, param=None):
        return super().query(text, param)


f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on_output()
    .on(Select.RecordCalls.query.args.text)
)

f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on(Select.RecordCalls.query.args.text)
    .on_output()
)

f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance")
    .on(Select.RecordCalls.query.args.text)
    .on_output()
    .aggregate(np.mean)
)


def evaluate_rag(rag_system: LightRAG, questions: list[str]):
    instrumented_rag = InstrumentedLightRAG(
        working_dir=rag_system.working_dir,
        llm_model_func=rag_system.llm_model_func,
        embedding_func=rag_system.embedding_func,
    )

    tru_rag = TruCustomApp(
        instrumented_rag,
        app_name="Islamic_Texts_RAG",
        app_version="base",
        feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
    )

    modes = ["naive", "local", "global", "hybrid"]

    with tru_rag as recording:
        for question in questions:
            for mode in modes:
                instrumented_rag.query(question, param=QueryParam(mode=mode))

    leaderboard = session.get_leaderboard()
    print("\nEvaluation Results:")
    print(leaderboard)

    leaderboard.to_csv("rag_evaluation_results.csv")
    print("\nResults saved to rag_evaluation_results.csv")

    return leaderboard


if __name__ == "__main__":
    from main import ISLAMIC_QUESTIONS, rag

    evaluation_questions = ISLAMIC_QUESTIONS + [
        "What is the Sunni path?",
        "Why are pride and arrogance considered dangerous in Islam?",
        "Is tobacco-smoking permissible in Islam?",
    ]

    results = evaluate_rag(rag, evaluation_questions)
