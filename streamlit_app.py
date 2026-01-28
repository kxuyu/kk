import os
import sys
import streamlit as st

# 1. 导入加载环境变量的库
from dotenv import load_dotenv, find_dotenv

# 2. 导入 LangChain 组件
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_chroma import Chroma  # 推荐使用新版 import

# 3. 导入你自定义的智谱组件
# 如果这两个文件在当前同级目录，直接 import 即可，不需要 sys.path.append
# 如果报错找不到模块，请确保这两个文件确实在代码运行的目录下
try:
    from zhipuai_embedding import ZhipuAIEmbeddings
    from zhipuai_llm import ZhipuaiLLM
except ImportError:
    st.error("❌ 找不到 zhipuai_embedding 或 zhipuai_llm 模块，请检查文件位置。")
    st.stop()

# --- 关键修复：加载环境变量 ---
# 这行代码会去寻找 .env 文件并加载 ZHIPUAI_API_KEY
_ = load_dotenv(find_dotenv())

# 关闭 Chroma 遥测
os.environ["ANONYMIZED_TELEMETRY"] = "False"

def get_retriever():
    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings()
    
    # 向量数据库持久化路径
    # ⚠️ 请确认这个路径下真的有 create_db.py 生成的数据
    persist_directory = '../data_base/vector_db/chroma' 
    # 或者用绝对路径更稳妥：
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # persist_directory = os.path.join(current_dir, "../data_base/vector_db/chroma")

    if not os.path.exists(persist_directory):
        st.error(f"找不到向量库路径: {persist_directory}")
        st.stop()

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    return vectordb.as_retriever(search_kwargs={"k": 3})

def combine_docs(docs):
    # 兼容性处理：有时候 docs 是 list，有时候可能是 dict
    if isinstance(docs, dict):
        return "\n\n".join(doc.page_content for doc in docs["context"])
    return "\n\n".join(doc.page_content for doc in docs)

# 使用 @st.cache_resource 缓存链的加载，避免每次提问都重新初始化模型
@st.cache_resource 
def get_qa_history_chain():
    retriever = get_retriever()
    
    # --- 关键修复：使用智谱 LLM 替换 ChatOpenAI ---
    api_key = os.environ.get("ZHIPUAI_API_KEY")
    llm = ZhipuaiLLM(
        model_name="glm-4-plus", 
        temperature=0.1, 
        api_key=api_key
    )
    
    # 1. 历史记录改写链
    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
        "不要回答问题，只需重写。"
    )
    condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    # 2. 问答链
    system_prompt = (
        "你是一个问答任务的助手。 "
        "请优先使用下方的【上下文信息】来回答用户的问题。 "
        "如果【上下文信息】中没有答案，或者上下文为空，请使用你自己的通用知识来回答用户。 "
        "请使用简洁的话语回答用户。"
        "\n\n"
        "上下文：{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context = retrieve_docs, 
    ).assign(answer=qa_chain)
    
    return qa_history_chain

def gen_response(chain, input, chat_history):
    # 调用链
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    
    # 解析流式输出
    # 注意：根据你的 Chain 结构，流式返回的可能是 dict chunks
    for res in response:
        if isinstance(res, dict) and "answer" in res:
            yield res["answer"]
        elif isinstance(res, str):
            yield res

def main():
    st.set_page_config(page_title="大模型知识库应用", page_icon="🦜")
    st.markdown('### 🦜🔗 动手学大模型应用开发 (基于智谱 GLM-4)')
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # 初始化链
    chain = get_qa_history_chain()
    
    messages = st.container(height=550)
    
    for message in st.session_state.messages:
        with messages.chat_message(message[0]):
            st.write(message[1])
            
    if prompt := st.chat_input("请输入问题..."):
        st.session_state.messages.append(("human", prompt))
        with messages.chat_message("human"):
            st.write(prompt)
            
        with messages.chat_message("ai"):
            # 将 list 格式的历史记录转为 LangChain 需要的格式（如果需要）
            # 这里简化处理，直接传 list
            answer_generator = gen_response(
                chain=chain,
                input=prompt,
                chat_history=st.session_state.messages # 注意：这里传入所有历史
            )
            output = st.write_stream(answer_generator)
            
        st.session_state.messages.append(("ai", output))

if __name__ == "__main__":
    main()
