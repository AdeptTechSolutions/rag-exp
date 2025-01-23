import os

import colorama
import numpy as np
import textract
from colorama import Fore, Style
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.llm import (
    gpt_4o_mini_complete,
    openai_complete_if_cache,
    openai_embedding,
)
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

load_dotenv(dotenv_path="../.env")

WORKING_DIR = "./processed"
BOOKS_DIR = "../data_mini"
PROCESS_FLAG = os.path.join(WORKING_DIR, ".processed")
ISLAMIC_QUESTIONS = [
    "What is the Sunni path?",
    "Why are pride and arrogance considered dangerous in Islam?",
    "Is tobacco-smoking permissible in Islam?",
]


def process_documents_folder(folder_path):
    """Extract text from all supported files in a directory"""
    all_text = ""
    supported_extensions = (".pdf", ".txt")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_extensions):
            file_path = os.path.join(folder_path, filename)
            try:
                text_content = textract.process(file_path)
                all_text += text_content.decode("utf-8") + "\n\n"
                print(f"Successfully processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    return all_text


if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gemini-1.5-flash",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model="text-embedding-004",
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


# quantization_config = BitsAndBytesConfig(load_in_4bit=True)
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=768, max_token_size=2048, func=embedding_func
    ),
    # llm_model_kwargs={"quantization_config": quantization_config, "device_map": "auto"},
    # addon_params={"insert_batch_size": 20},
    # llm_model_max_async=30,
)

colorama.init()

if os.path.exists(BOOKS_DIR):
    if not os.path.exists(PROCESS_FLAG):
        islamic_texts = process_documents_folder(BOOKS_DIR)
        if islamic_texts:
            rag.insert(islamic_texts)
            with open(PROCESS_FLAG, "w") as f:
                f.write("processed")
            print("\nSuccessfully loaded Islamic texts into RAG!\n")
        else:
            print("No processable texts found. Exiting.")
            exit()
    else:
        print("\nDocuments already processed. Skipping insertion.\n")
else:
    print(f"Books directory '{BOOKS_DIR}' not found. Exiting.")
    exit()


for question in ISLAMIC_QUESTIONS:
    print(f"\n{Fore.GREEN}Question:{Style.RESET_ALL} {question}")

    for mode in ["naive", "local", "global", "hybrid"]:
        # context = rag.query(
        #     question, param=QueryParam(mode=mode, only_need_context=True)
        # )
        answer = rag.query(question, param=QueryParam(mode=mode))

        print(f"\n{Fore.BLUE}Mode:{Style.RESET_ALL} {mode}")
        # print(f"Context: {context}")
        print(f"Answer: {answer}")

    print("-" * 80)
