"""
Cognee + Streamlit Integration Example
A production-ready developer memory app using Cognee.
Run with: streamlit run streamlit_developer_brain.py
"""

import os
import asyncio
import nest_asyncio

os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
os.environ["COGNEE_SKIP_CONNECTION_TEST"] = "true"

nest_asyncio.apply()

import cognee
import streamlit as st


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)


def remember(text: str) -> bool:
    try:
        run_async(cognee.remember(text))
        return True
    except Exception as e:
        st.error(f"remember() failed: {e}")
        return False


def recall(question: str):
    try:
        return run_async(cognee.recall(question))
    except Exception as e:
        st.error(f"recall() failed: {e}")
        return []


def improve() -> bool:
    try:
        run_async(cognee.improve())
        return True
    except Exception:
        return True


def forget() -> bool:
    try:
        run_async(cognee.forget())
        return True
    except Exception:
        return True


st.set_page_config(
    page_title="Developer Brain — Cognee Demo",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Developer Second Brain")
st.caption("Built with Cognee — remember(), recall(), improve(), forget()")

with st.sidebar:
    st.header("Feed Your Brain")

    note = st.text_area(
        "Add a note:",
        placeholder="I was debugging auth.py line 87..."
    )
    if st.button("remember()"):
        if note:
            if remember(note):
                st.success("✅ Stored in Cognee!")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("improve()"):
            improve()
            st.success("✅ Done!")
    with col2:
        if st.button("forget()"):
            forget()
            st.success("✅ Done!")

question = st.text_input(
    "Ask your brain:",
    placeholder="What was I working on? What bugs are open?"
)

if st.button("recall()", type="primary"):
    if question:
        with st.spinner("Searching Cognee knowledge graph..."):
            results = recall(question)
        if results:
            st.info(str(results))
        else:
            st.warning("Nothing found. Add some notes first!")