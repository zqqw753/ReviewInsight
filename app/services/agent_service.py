from typing import TypedDict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, END, START
from app.core.config import settings
from app.services.vector_service import search_comments

class AgentState(TypedDict):
    question: str
    question_type: str
    context: str
    sources: List[str]
    answer: str

llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=settings.deepseek_api_key,
    temperature=0.3
)

def classify_question(state: AgentState) :
    question = state["question"]

    prompt = ChatPromptTemplate.from_messages([
        ("system","你是一个分类助手。请判断用户的问题属于以下哪种类型：'缺点'、'优点'、'整体评价'。只输出这三个词中的一个，\
        不要输出任何标点符号或其他内容。"),
        ("user", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()
    q_type = chain.invoke({"question": question}).strip()

    if q_type not in ['缺点', '优点', '整体评价']:
        q_type = "整体评价"

    print(f"用户的问题类型为{q_type}")
    return {"question_type": q_type}

def retrieve_documents(state: AgentState):
    question = state["question"]

    contexts, source = search_comments(question, 10)
    context_str = "\n\n".join(contexts)

    return{
        "context": context_str,
        "sources": source
    }

def analyze_defect(state: AgentState):
    prompt = ChatPromptTemplate([
        ("system", "你是一名分析师。请根据评论片段，找出产品的所有缺点，并以列表形式列出，估算频次。"),
        ("user", "参考评论：\n{context}\n\n用户问题：{question}")
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": state["context"], "question": state["question"]})

    return{"answer": answer}

def analyze_merit(state: AgentState):
    prompt = ChatPromptTemplate([
        ("system", "你是一名分析师。请根据评论片段，找出产品的所有优点，并以列表形式列出，估算频次。"),
        ("user", "参考评论：\n{context}\n\n用户问题：{question}")
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": state["context"], "question": state["question"]})

    return{"answer": answer}

def analyze_overall(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一名分析师。请根据评论片段，给出正面/负面/中性的大致比例，并总结整体评价。"),
        ("user", "参考评论：\n{context}\n\n用户问题：{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": state["context"], "question": state["question"]})

    return {"answer": answer}

def route_question(state: AgentState):
    return state["question_type"]

def build_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_question", classify_question)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("analyze_defect", analyze_defect)
    workflow.add_node("analyze_merit", analyze_merit)
    workflow.add_node("analyze_overall", analyze_overall)

    workflow.add_edge(START, "classify_question")
    workflow.add_edge("classify_question", "retrieve_documents")
    workflow.add_conditional_edges("retrieve_documents",
                                   route_question,
                                   path_map={"缺点":"analyze_defect",
                                             "优点":"analyze_merit",
                                             "整体评价":"analyze_overall"})
    workflow.add_edge("analyze_defect", END)
    workflow.add_edge("analyze_merit", END)
    workflow.add_edge("analyze_overall", END)

    app = workflow.compile()

    return app

agent_app = build_agent()