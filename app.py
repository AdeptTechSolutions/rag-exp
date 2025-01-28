import os
from pathlib import Path

import fitz
import streamlit as st
from PIL import Image

from core import RAGModel

st.set_page_config(
    page_title="Islamic Texts",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def load_logo():
    try:
        return Image.open("resources/_.png")
    except Exception:
        return None


def get_pdf_list():
    data_dir = Path("data")
    return sorted([f.name for f in data_dir.glob("*.pdf")])


def display_pdf_page(pdf_path: str, page_num: int):
    try:
        doc = fitz.open(f"data/{pdf_path}")
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

        img_path = "temp_page.png"
        pix.save(img_path)

        st.image(img_path, width=600)

        os.remove(img_path)
        doc.close()
    except Exception as e:
        st.error(f"Error displaying PDF page: {str(e)}")


def main():
    st.markdown(
        "<h1 style='text-align: center;'>📚 Islamic Texts</h1>",
        unsafe_allow_html=True,
    )

    st.write("")

    with st.form("query_form"):
        query = st.text_input(
            "Enter your question:",
            placeholder="Ask a question about the documents...",
        )
        submit_button = st.form_submit_button("Generate response")

    if submit_button and query:
        with st.spinner("Generating response..."):
            model = RAGModel()
            answer, source = model.get_answer(query)

            st.write("#### Answer")
            st.write(answer)

            st.write("")

            with st.expander("📑 Source Information", expanded=False):
                if source and source["source"] and source["page"]:
                    st.info(
                        """
                    ```
                    Document: {}
                    Page: {}
                    Relevance Score: {:.4f}
                    ```
                    """.format(
                            source["source"], source["page"], source["score"]
                        )
                    )

                    st.info(
                        """
                        **Context**
                        """
                    )
                    st.code(source["content"])

                    st.info(
                        """
                        **Page Preview**
                        """
                    )

                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        display_pdf_page(source["source"], source["page"])
                else:
                    st.warning("⚠️ No specific source found for this response.")


if __name__ == "__main__":
    main()
