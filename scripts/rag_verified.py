#!/usr/bin/env python3
# ==============================================================================
# WEDA LLM Suite — verified RAG (kind: rag-verified)
# ==============================================================================
# A LangGraph pipeline around the shared document store:
#   retrieve -> grade -> (rewrite -> retrieve)* -> answer -> verify -> done
# Each grade/verify step is one model call over all snippets at once, so a
# question costs 3-5 calls. Retrieval and generation go through the same
# functions the other apps use; the graph only decides what happens next.
import json
import re
import sys
import os
from typing import List, TypedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.loader import AppConfig
import agent_demo
import rag_demo


class State(TypedDict, total=False):
    question: str
    hint: str
    query: str
    docs: List[str]
    srcs: List[str]
    relevant: List[int]
    answer: str
    verdict: str
    rounds: int
    strict: bool


def _llm(app, messages, max_tokens=512):
    body = agent_demo.chat_body(app, max_tokens=max_tokens)
    body["messages"] = messages
    resp, _ = agent_demo.post_chat(app.generation_base(), body)
    with resp:
        data = json.loads(resp.read())
    return (data["choices"][0]["message"].get("content") or "").strip()


def build(app: AppConfig, trace=None):
    """Compile the graph; `trace(message)` observes each step."""
    from langgraph.graph import StateGraph, END
    llm = lambda messages, max_tokens=512: _llm(app, messages, max_tokens)
    retrieve = lambda q, h, n: rag_demo.retrieve(app, q, h, n)
    say = trace or (lambda m: None)
    top_k = int(app.top_k or 4)
    max_rounds = int(app.verify_rounds or 1)

    def n_retrieve(state):
        got = retrieve(state["query"], state.get("hint", ""), top_k * 2)
        docs, srcs = got if got else ([], [])
        say(f"retrieve: {len(docs)} candidates for '{state['query']}'")
        return {"docs": docs, "srcs": srcs}

    def n_grade(state):
        if not state["docs"]:
            say("grade: nothing to grade")
            return {"relevant": []}
        listing = "\n\n".join(f"[{i+1}] {d[:700]}" for i, d in enumerate(state["docs"]))
        out = llm([
            {"role": "system", "content": "You grade search results. Reply with the numbers of the "
             "snippets that contain information useful for answering the question, comma-separated, "
             "or NONE. Reply with numbers only."},
            {"role": "user", "content": f"Question: {state['question']}\n\nSnippets:\n{listing}"}],
            max_tokens=64)
        nums = [int(x) - 1 for x in re.findall(r"\d+", out)]
        relevant = [i for i in nums if 0 <= i < len(state["docs"])][:top_k]
        say(f"grade: {len(relevant)} of {len(state['docs'])} relevant")
        return {"relevant": relevant}

    def after_grade(state):
        if state["relevant"]:
            return "answer"
        if state.get("rounds", 0) < max_rounds:
            return "rewrite"
        return "give_up"

    def n_rewrite(state):
        out = llm([
            {"role": "system", "content": "Rewrite the user's question as a short keyword search query "
             "for a document index. Reply with the query only."},
            {"role": "user", "content": state["question"]}], max_tokens=48)
        query = out.splitlines()[0].strip().strip('"') or state["question"]
        say(f"rewrite: '{query}'")
        return {"query": query, "rounds": state.get("rounds", 0) + 1}

    def n_answer(state):
        docs = [state["docs"][i] for i in state["relevant"]]
        srcs = [state["srcs"][i] for i in state["relevant"]]
        system = agent_demo.system_prompt_for(app) or rag_demo.DEFAULT_RAG_PROMPT
        if state.get("strict"):
            system += ("\nState only facts present in the context. Where the context is silent, "
                       "say that it does not say.")
        out = llm([{"role": "system", "content": system},
                   {"role": "user",
                    "content": rag_demo.rag_user_message(docs, srcs, state["question"])}],
                  max_tokens=768)
        say("answer: drafted" + (" (strict)" if state.get("strict") else ""))
        return {"answer": out, "srcs": srcs, "docs": docs}

    def n_verify(state):
        context = "\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(state["docs"]))
        out = llm([
            {"role": "system", "content": "You check an answer against its sources. If every factual "
             "claim in the answer is supported by the context, reply SUPPORTED. Otherwise reply "
             "UNSUPPORTED: followed by the unsupported claim. One line."},
            {"role": "user", "content": f"Context:\n{context}\n\nAnswer:\n{state['answer']}"}],
            max_tokens=96)
        verdict = "SUPPORTED" if out.upper().startswith("SUPPORTED") else out
        say(f"verify: {verdict[:80]}")
        return {"verdict": verdict}

    def after_verify(state):
        if state["verdict"] == "SUPPORTED" or state.get("strict"):
            return END
        return "strict_answer"

    def n_strict(state):
        say("verify failed: answering again strictly")
        return {"strict": True}

    def n_give_up(state):
        say("no relevant snippets after retry: declining")
        return {"answer": "The documents do not contain information to answer this question.",
                "srcs": [], "docs": [], "verdict": "SUPPORTED"}

    g = StateGraph(State)
    g.add_node("retrieve", n_retrieve)
    g.add_node("grade", n_grade)
    g.add_node("rewrite", n_rewrite)
    g.add_node("answer", n_answer)
    g.add_node("verify", n_verify)
    g.add_node("strict_answer", n_strict)
    g.add_node("give_up", n_give_up)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade,
                            {"answer": "answer", "rewrite": "rewrite", "give_up": "give_up"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("answer", "verify")
    g.add_conditional_edges("verify", after_verify, {END: END, "strict_answer": "strict_answer"})
    g.add_edge("strict_answer", "answer")
    g.add_edge("give_up", END)
    return g.compile()


def answer(app: AppConfig, question: str, hint: str = "", trace=None):
    """Run the pipeline; returns {answer, sources}."""
    graph = build(app, trace=trace)
    final = graph.invoke({"question": question, "hint": hint, "query": question,
                          "rounds": 0, "strict": False})
    text = final.get("answer", "")
    if final.get("verdict") and final["verdict"] != "SUPPORTED":
        text += ("\n\n(verification: a statement above could not be confirmed from the "
                 f"documents -- {final['verdict'].replace('UNSUPPORTED:', '').strip()})")
    return {"answer": text, "sources": sorted(set(final.get("srcs", [])))}
