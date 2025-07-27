# 配置日志
import logging
import os
import asyncio
import time
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from typing_extensions import TypedDict, Annotated

# LangGraph相关导入
from langgraph.graph import StateGraph, START, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# LangChain相关导入
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# RAG相关导入
# from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# 内存管理工具导入
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

# MCP客户端（假设您有这个导入）
from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain.embeddings.base import Embeddings
import requests
# from typing import List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    """消息类型枚举"""
    HUMAN = "human"
    AI = "ai"
    SYSTEM = "system"
    ERROR = "error"

@dataclass
class Config:
    """配置管理类"""
    MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "http://192.168.100.99:30088/sse")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "sk-f")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")  # 用于向量化
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    NAMESPACE: str = os.getenv("K8S_NAMESPACE", "kubernetes")
    
    # RAG相关配置
    KNOWLEDGE_BASE_PATH: str = os.getenv("KNOWLEDGE_BASE_PATH", "./knowledge_base")
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./vector_store")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    
    # 内存管理配置
    MEMORY_NAMESPACE: str = os.getenv("MEMORY_NAMESPACE", "kubernetes-bot")
    
    def __post_init__(self):
        """配置验证"""
        if not self.DEEPSEEK_API_KEY:
            logger.warning("DEEPSEEK_API_KEY not set, may cause authentication issues")
        
        if not self.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set, using alternative embedding method")
        
        # 设置日志级别
        logging.getLogger().setLevel(getattr(logging, self.LOG_LEVEL.upper()))

class State(TypedDict):
    """状态定义"""
    messages: Annotated[list, add_messages]
    user_context: Optional[Dict[str, Any]]
    last_tool_call: Optional[str]
    retrieved_docs: Optional[List[Document]]  # RAG检索到的文档
    use_rag: bool  # 是否使用RAG
    memory_results: Optional[List[Dict[str, Any]]]  # 内存搜索结果



class SiliconFlowEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "Qwen/Qwen3-Embedding-8B"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.siliconflow.cn/v1/embeddings"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model,
            "input": text
        }
        response = requests.post(self.api_url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]

class RAGManager:
    """RAG知识库管理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.vectorstore = None
        self.retriever = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        
    async def initialize(self):
        """初始化RAG系统"""
        try:
            # 初始化embeddings
            if self.config.OPENAI_API_KEY:
                embeddings = OpenAIEmbeddings(
                    openai_api_key=self.config.OPENAI_API_KEY
                )
            else:
                # 使用替代方案，比如本地embedding模型
                logger.warning("Using alternative embedding method")
                # from langchain_openai import OpenAIEmbeddings
                # embeddings = OpenAIEmbeddings(
                #     openai_api_key="sk-a",
                #     openai_api_base="https://api.siliconflow.cn/v1/embeddings",  # 替换为你的服务地址
                #     model="Qwen/Qwen3-Embedding-8B"
                # )    
                embeddings = SiliconFlowEmbeddings(api_key="sk-")

            
            # 检查是否已有向量库
            if os.path.exists(self.config.VECTOR_STORE_PATH):
                logger.info("Loading existing Chroma vector store")
                self.vectorstore = Chroma(
                    embedding_function=embeddings,
                    persist_directory=self.config.VECTOR_STORE_PATH
                )
            else:
                logger.info("Creating new Chroma vector store")
                await self._build_knowledge_base(embeddings)
            
            # 设置检索器
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.TOP_K_RETRIEVAL}
            )
            
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def _get_default_k8s_knowledge(self) -> List[Document]:
        """获取默认Kubernetes知识"""
        default_docs = [
            Document(
                page_content="""
Kubernetes节点调度和资源管理:

1. 节点状态检查:
   - Ready: 节点健康且可以接受Pod
   - NotReady: 节点有问题，不能调度新Pod
   - Unknown: 节点状态未知

2. 资源分配原则:
   - CPU和内存请求(requests)用于调度决策
   - 限制(limits)用于运行时资源控制
   - 节点可分配资源 = 总资源 - 系统预留 - kubelet预留

3. 调度约束:
   - nodeSelector: 基于标签选择节点
   - affinity/anti-affinity: 更复杂的调度规则
   - taints和tolerations: 节点排斥和容忍机制
                """,
                metadata={"source": "k8s_scheduling", "type": "knowledge"}
            ),
            Document(
                page_content="""
Kubernetes故障诊断指南:

1. Pod无法调度问题:
   - 检查节点资源是否充足
   - 验证调度约束是否满足
   - 查看事件日志获取详细信息

2. 节点问题排查:
   - 检查节点状态和条件
   - 验证kubelet服务状态
   - 查看系统资源使用情况

3. 常用诊断命令:
   - kubectl describe node <node-name>
   - kubectl describe pod <pod-name>
   - kubectl get events --sort-by=.metadata.creationTimestamp
                """,
                metadata={"source": "k8s_troubleshooting", "type": "knowledge"}
            ),
            Document(
                page_content="""
Kubernetes最佳实践:

1. 资源管理:
   - 始终设置资源请求和限制
   - 使用命名空间隔离不同环境
   - 定期清理未使用的资源

2. 安全实践:
   - 使用RBAC控制访问权限
   - 定期更新镜像和组件
   - 启用网络策略

3. 监控和日志:
   - 部署监控系统(如Prometheus)
   - 集中化日志管理
   - 设置适当的告警规则
                """,
                metadata={"source": "k8s_best_practices", "type": "knowledge"}
            )
        ]
        return default_docs
    
    async def retrieve_relevant_docs(self, query: str) -> List[Document]:
        """检索相关文档"""
        if not self.retriever:
            logger.warning("RAG retriever not initialized")
            return []
        
        try:
            docs = await self.retriever.ainvoke(query)
            logger.info(f"Retrieved {len(docs)} relevant documents")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    async def _build_knowledge_base(self, embeddings):
        """构建知识库"""
        try:
            documents = []

            # 加载知识库文档
            if os.path.exists(self.config.KNOWLEDGE_BASE_PATH):
                loader = DirectoryLoader(
                    self.config.KNOWLEDGE_BASE_PATH,
                    glob="**/*.txt",
                    loader_cls=TextLoader,
                    show_progress=True
                )
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {self.config.KNOWLEDGE_BASE_PATH}")
            else:
                logger.warning(f"Knowledge base path not found: {self.config.KNOWLEDGE_BASE_PATH}")

            # 添加默认 Kubernetes 知识
            default_k8s_docs = self._get_default_k8s_knowledge()
            documents.extend(default_k8s_docs)

            if not documents:
                logger.warning("No documents found for knowledge base")
                documents = [Document(page_content="Empty knowledge base", metadata={})]

            # ✅ 这里包在 try 内
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Split into {len(split_docs)} chunks")

            # ✅ 构建 Chroma 向量库
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=embeddings,
                persist_directory=self.config.VECTOR_STORE_PATH
            )
            self.vectorstore.persist()

            logger.info("Knowledge base built and saved successfully")

        except Exception as e:
            logger.error(f"Failed to build knowledge base: {e}")
            raise

class KubernetesToolManager:
    """Kubernetes工具管理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = None
        self.tools = []
        
        # 初始化内存存储和工具
        self.memory_store = InMemoryStore()
        self.manage_tool = create_manage_memory_tool(
            store=self.memory_store, 
                namespace=(
                    "Kubernetes_bot",
                    "{langgraph_user_id}",
                    "collection"
                )
        )
        self.search_tool = create_search_memory_tool(
            store=self.memory_store, 
                namespace=(
                    "Kubernetes_bot",
                    "{langgraph_user_id}",
                    "collection"
                )
        )
    
    @classmethod
    async def create(cls, config: Config):
        self = cls(config)
        """初始化工具连接"""
        try:
            logger.info(f"Connecting to MCP server at {self.config.MCP_SERVER_URL}")
            # 注释掉实际的MCP连接，使用模拟工具
            # self.client = MultiServerMCPClient({
            #     "kubernetes": {
            #     "url": self.config.MCP_SERVER_URL,
            #     "transport": "sse"
            #     }
            # })
            
            # self.tools = await self.client.get_tools()
            
            # 模拟工具列表（包含内存管理工具）
            self.tools = self._get_mock_tools()
            
            # 添加内存管理工具
            self.tools.extend([self.manage_tool, self.search_tool])
            
            logger.info(f"Successfully loaded {len(self.tools)} tools (including memory tools)")
            
            # 打印可用工具
            for tool in self.tools:
                tool_name = getattr(tool, 'name', 'unknown_tool')
                logger.debug(f"Available tool: {tool_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")
            # 不抛出异常，使用模拟工具和内存工具
            self.tools = self._get_mock_tools()
            self.tools.extend([self.manage_tool, self.search_tool])
        return self
    
    def _get_mock_tools(self):
        """获取模拟工具"""
        # 这里返回空列表，实际使用时替换为真实工具
        return []
    
    def get_tool(self):
        """获取工具列表"""
        return self.tools
    
    async def save_to_memory(self, key: str, value: Any, metadata: Dict[str, Any] = None):
        """保存信息到内存"""
        try:
            await self.manage_tool.ainvoke({
                "action": "create", 
                "key": key, 
                "value": value,
                "metadata": metadata or {}
            })
            logger.info(f"Saved to memory: {key}")
        except Exception as e:
            logger.error(f"Failed to save to memory: {e}")
    
    async def search_memory(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索内存"""
        try:
            result = await self.search_tool.ainvoke({
                "query": query,
                "limit": limit
            })
            logger.info(f"Memory search returned {len(result) if result else 0} results")
            return result if result else []
        except Exception as e:
            logger.error(f"Failed to search memory: {e}")
            return []
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            return True  # 模拟健康状态
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

class KubernetesChatBot:
    """Kubernetes聊天机器人"""
    
    def __init__(self, config: Config, tool_manager, rag_manager, llm, llm_with_tools, graph):
        self.config = config
        self.tool_manager = tool_manager
        self.rag_manager = rag_manager
        self.llm = llm
        self.llm_with_tools = llm_with_tools
        self.graph = graph

    @classmethod
    async def create(cls, config: Config):
        """异步构造函数"""
        # 初始化RAG管理器
        rag_manager = RAGManager(config)
        await rag_manager.initialize()
        
        # 初始化工具
        tool_manager = await KubernetesToolManager.create(config)

        # 初始化 LLM
        llm = ChatDeepSeek(
            model=config.DEEPSEEK_MODEL,
            api_key=config.DEEPSEEK_API_KEY,
        )
        logger.info("LLM initialized successfully")

        tools = tool_manager.get_tool()

        # 包装工具
        llm_with_tools = llm.bind_tools(tools) if tools else llm

        def build_rag_node(rag_manager, tool_manager):
            """构建RAG节点"""
            async def rag_node(state: State) -> Dict[str, Any]:
                messages = state["messages"]
                if not messages:
                    return {"retrieved_docs": [], "memory_results": []}
                
                # 获取最新的用户消息
                last_message = messages[-1]
                if hasattr(last_message, 'content'):
                    query = last_message.content
                else:
                    query = str(last_message)
                
                logger.info(f"RAG retrieving for query: {query[:100]}...")
                
                # 检索相关文档
                retrieved_docs = await rag_manager.retrieve_relevant_docs(query)
                
                # 搜索内存中的相关信息
                # memory_results = await tool_manager.search_memory(query, limit=5)
                
                return {
                    "retrieved_docs": retrieved_docs,
                    # "memory_results": memory_results,
                    "use_rag": len(retrieved_docs) > 0 or len(memory_results) > 0
                }
            
            return rag_node

        def build_chatbot_node(llm_with_tools, config, tool_manager):
            """构建聊天机器人节点"""
            async def chatbot_node(state: State) -> Dict[str, Any]:
                retry_count = 0
                last_exception = None

                while retry_count < config.MAX_RETRIES:
                    try:
                        messages = state["messages"]
                        retrieved_docs = state.get("retrieved_docs", [])
                        # memory_results = state.get("memory_results", [])
                        
                        # 准备系统提示，包含RAG和内存信息
                        system_prompt = cls._build_system_prompt(retrieved_docs)
                        
                        # 准备消息 - FIXED: Better message handling
                        formatted_messages = [SystemMessage(content=system_prompt)]
                        
                        for msg in messages:
                            content = ""
                            
                            # Handle different message types properly
                            if hasattr(msg, 'content'):
                                content = msg.content
                            elif isinstance(msg, dict):
                                if 'content' in msg:
                                    content = msg['content']
                                elif 'message' in msg:
                                    content = msg['message']
                                else:
                                    # Try to get content from dict using .get() safely
                                    content = str(msg)
                            elif isinstance(msg, str):
                                content = msg
                            else:
                                content = str(msg)
                            
                            # Only add non-empty messages
                            if content.strip():
                                formatted_messages.append(HumanMessage(content=content))

                        logger.info(f"Processing request (attempt {retry_count + 1}) with {len(formatted_messages)} messages")
                        start_time = time.time()

                        response = await llm_with_tools.ainvoke(formatted_messages)

                        duration = time.time() - start_time
                        logger.info(f"Request processed in {duration:.2f}s")

                        return {
                            "messages": [response],
                            "last_tool_call": None
                        }

                    except Exception as e:
                        retry_count += 1
                        last_exception = e
                        logger.error(f"Chatbot error (attempt {retry_count}): {e}")
                        logger.error(f"Exception type: {type(e).__name__}")
                        logger.error(f"State structure: {type(state)}, keys: {state.keys() if hasattr(state, 'keys') else 'N/A'}")
                        
                        # Log the problematic message structure for debugging
                        if "messages" in state:
                            for i, msg in enumerate(state["messages"]):
                                logger.error(f"Message {i}: type={type(msg)}, content={str(msg)[:100]}...")

                        if retry_count < config.MAX_RETRIES:
                            await asyncio.sleep(2 ** retry_count)

                return {
                    "messages": [AIMessage(content="抱歉，系统暂时无法处理您的请求，请稍后重试。")],
                    "error": str(last_exception) if last_exception else "Unknown error"
                }

            return chatbot_node
        # def build_chatbot_node(llm_with_tools, config, tool_manager):
        #     """构建聊天机器人节点"""
        #     async def chatbot_node(state: State) -> Dict[str, Any]:
        #         retry_count = 0
        #         last_exception = None

        #         while retry_count < config.MAX_RETRIES:
        #             try:
        #                 messages = state["messages"]
        #                 retrieved_docs = state.get("retrieved_docs", [])
        #                 memory_results = state.get("memory_results", [])
                        
        #                 # 准备系统提示，包含RAG和内存信息
        #                 system_prompt = cls._build_system_prompt(retrieved_docs, memory_results)
                        
        #                 # 准备消息
        #                 formatted_messages = [SystemMessage(content=system_prompt)]
        #                 for msg in messages:
        #                     if hasattr(msg, 'content'):
        #                         formatted_messages.append(HumanMessage(content=msg.content))
        #                     else:
        #                         formatted_messages.append(HumanMessage(content=str(msg)))

        #                 logger.info(f"Processing request (attempt {retry_count + 1})")
        #                 start_time = time.time()

        #                 response = await llm_with_tools.ainvoke(formatted_messages)

        #                 duration = time.time() - start_time
        #                 logger.info(f"Request processed in {duration:.2f}s")

        #                 return {
        #                     "messages": [response],
        #                     "last_tool_call": None
        #                 }

        #             except Exception as e:
        #                 retry_count += 1
        #                 last_exception = e
        #                 logger.error(f"Chatbot error (attempt {retry_count}): {e}")

        #                 if retry_count < config.MAX_RETRIES:
        #                     await asyncio.sleep(2 ** retry_count)

        #         return {
        #             "messages": [AIMessage(content="抱歉，系统暂时无法处理您的请求，请稍后重试。")]
        #         }

        #     return chatbot_node

        # 构建节点
        rag_node = build_rag_node(rag_manager, tool_manager)
        chatbot_node = build_chatbot_node(llm_with_tools, config, tool_manager)

        # 构建图
        graph_builder = StateGraph(State)
        graph_builder.add_node("rag", rag_node)
        graph_builder.add_node("chatbot", chatbot_node)
        
        if tools:  # 只有在有工具时才添加工具节点
            graph_builder.add_node("tools", ToolNode(tools=tools))
        
        # 设置边
        graph_builder.add_edge(START, "rag")
        graph_builder.add_edge("rag", "chatbot")
        
        if tools:
            graph_builder.add_conditional_edges("chatbot", tools_condition)
            graph_builder.add_edge("tools", "chatbot")
        
        memory = MemorySaver()
        graph = graph_builder.compile(checkpointer=memory)

        return cls(config, tool_manager, rag_manager, llm, llm_with_tools, graph)
    
    @staticmethod
    def _build_system_prompt(retrieved_docs: List[Document]) -> str:
        """构建包含RAG信息和内存信息的系统提示"""
        base_prompt = """你是一个专业的Kubernetes管理助手。你的任务是帮助用户管理和查询Kubernetes集群。

请遵循以下原则：
1. 优先使用提供的工具来获取实时的集群信息
2. 结合知识库信息和历史记录提供准确、详细的信息和建议
3. 如果操作可能有风险，请明确警告用户
4. 使用清晰、易懂的语言回答
5. 如果不确定，请说明不确定性

当前可用的工具可以帮你：
- 查询节点状态和资源使用情况
- 检查Pod调度状态
- 分析集群健康状况
- 提供运维建议
- 管理和搜索对话记录
"""

        if retrieved_docs:
            knowledge_context = "\n\n=== 相关知识库信息 ===\n"
            for i, doc in enumerate(retrieved_docs, 1):
                knowledge_context += f"\n{i}. {doc.page_content[:500]}...\n"
            base_prompt += knowledge_context
        
        if retrieved_docs:
            base_prompt += "\n请结合以上信息来回答用户问题。"
        
        return base_prompt

    async def chat(self, user_input: str, thread_id: str = "default") -> None:
        """执行对话"""
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            logger.info(f"Starting chat session: {thread_id}")
            
            async for event in self.graph.astream(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "user_context": {"timestamp": time.time()},
                    "use_rag": True
                },
                config=config,
                stream_mode="values",
            ):
                if "messages" in event and event["messages"]:
                    last_message = event["messages"][-1]
                    if hasattr(last_message, 'pretty_print'):
                        last_message.pretty_print()
                    else:
                        print(f"Response: {last_message}")
                        
                # 显示RAG和内存信息
                if "retrieved_docs" in event and event["retrieved_docs"]:
                    print(f"\n[RAG] 检索到 {len(event['retrieved_docs'])} 个相关文档")
                
                if "memory_results" in event and event["memory_results"]:
                    print(f"[Memory] 找到 {len(event['memory_results'])} 个相关历史记录")
        
        except Exception as e:
            logger.error(f"Chat execution failed: {e}")
            print(f"聊天执行失败: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """系统健康检查"""
        status = {
            "llm_status": "unknown",
            "tools_status": "unknown", 
            "rag_status": "unknown",
            "memory_status": "unknown",
            "overall_status": "unknown"
        }
        
        try:
            # 检查LLM
            test_response = await self.llm.ainvoke([HumanMessage(content="健康检查")])
            status["llm_status"] = "healthy"
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            status["llm_status"] = "unhealthy"
        
        # 检查工具
        status["tools_status"] = "healthy" if self.tool_manager.health_check() else "unhealthy"
        
        # 检查RAG
        status["rag_status"] = "healthy" if self.rag_manager.retriever else "unhealthy"
        
        # 检查内存工具
        try:
            # 测试内存工具
            test_key = f"health_check_{int(time.time())}"
            await self.tool_manager.save_to_memory(test_key, "test_value")
            memory_search_result = await self.tool_manager.search_memory("health_check", limit=1)
            status["memory_status"] = "healthy"
        except Exception as e:
            logger.error(f"Memory tools health check failed: {e}")
            status["memory_status"] = "unhealthy"
        
        # 总体状态
        healthy_components = sum(1 for s in [
            status["llm_status"], 
            status["tools_status"], 
            status["rag_status"], 
            status["memory_status"]
        ] if s == "healthy")
        
        if healthy_components >= 3:
            status["overall_status"] = "healthy"
        else:
            status["overall_status"] = "unhealthy"
        
        return status

    async def add_knowledge(self, content: str, metadata: Dict[str, Any] = None) -> bool:
        """动态添加知识到向量库"""
        try:
            doc = Document(page_content=content, metadata=metadata or {})
            split_docs = self.rag_manager.text_splitter.split_documents([doc])
            
            if self.rag_manager.vectorstore:
                self.rag_manager.vectorstore.add_documents(split_docs)
                logger.info(f"Added {len(split_docs)} chunks to knowledge base")
                return True
            else:
                logger.error("Vector store not initialized")
                return False
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}")
            return False

    async def search_memory_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """根据主题搜索内存中的对话记录"""
        return await self.tool_manager.search_memory(topic, limit=10)
    
    async def get_conversation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取最近的对话历史"""
        return await self.tool_manager.search_memory("conversation", limit=limit)

async def main():
    """主函数"""
    try:
        # 初始化配置
        config = Config()
        
        # 创建聊天机器人
        chatbot = await KubernetesChatBot.create(config)
        
        # 健康检查
        health = await chatbot.health_check()
        logger.info(f"System health: {health}")
        
        if health["overall_status"] != "healthy":
            logger.warning("System is not fully healthy, some features may not work correctly")
        
        # 执行示例查询
        user_input = "请帮我查询 node-1 的调度状态，并判断是否可以分配 Pod abc-123。同时给出一些最佳实践建议。"
        
        print("=== Kubernetes ChatBot with RAG ===")
        print(f"用户输入: {user_input}")
        print("\n机器人回复:")
        
        await chatbot.chat(user_input, "kube-rag-session-1")
        
        # 演示动态添加知识
        print("\n=== 添加新知识 ===")
        new_knowledge = """
        Kubernetes Pod故障排查新方法:
        1. 使用kubectl logs检查容器日志
        2. 使用kubectl exec进入容器调试
        3. 检查Pod的资源使用情况
        4. 验证镜像拉取策略
        """
        
        success = await chatbot.add_knowledge(
            new_knowledge, 
            {"source": "dynamic_add", "timestamp": time.time()}
        )
        
        if success:
            print("新知识添加成功！")
            
            # 测试新知识
            test_query = "Pod故障排查有什么新方法？"
            print(f"\n测试查询: {test_query}")
            await chatbot.chat(test_query, "kube-rag-session-2")
        
        # 演示内存搜索功能
        print("\n=== 内存搜索测试 ===")
        
        # 搜索之前的对话记录
        memory_results = await chatbot.search_memory_by_topic("node")
        print(f"找到 {len(memory_results)} 个与'node'相关的历史记录")
        
        # 获取对话历史
        history = await chatbot.get_conversation_history(limit=5)
        print(f"最近 {len(history)} 条对话历史:")
        print("history:",type(history))
        print("history0:",type(history[0]))
        # 测试相关性查询
        print("\n=== 相关性查询测试 ===")
        related_query = "之前我们讨论过什么关于Pod调度的问题吗？"
        await chatbot.chat(related_query, "kube-rag-session-3")
        
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        print(f"应用启动失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())