import logging
import asyncio
import os
import time
from typing import Annotated, Optional, Dict, Any
from typing_extensions import TypedDict
from dataclasses import dataclass
from enum import Enum

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import Tool

from langchain_deepseek import ChatDeepSeek
from langchain_mcp_adapters.client import MultiServerMCPClient

# 配置日志
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
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "sk-")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    #REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    NAMESPACE: str = os.getenv("K8S_NAMESPACE", "kubernetes")
    
    def __post_init__(self):
        """配置验证"""
        if not self.DEEPSEEK_API_KEY:
            logger.warning("DEEPSEEK_API_KEY not set, may cause authentication issues")
        
        # 设置日志级别
        logging.getLogger().setLevel(getattr(logging, self.LOG_LEVEL.upper()))

class State(TypedDict):
    """状态定义"""
    messages: Annotated[list, add_messages]
    user_context: Optional[Dict[str, Any]]
    last_tool_call: Optional[str]
 

class KubernetesToolManager:
    """Kubernetes工具管理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = None
        self.tools = []
    
    @classmethod
    async def create(cls, config: Config):
        self = cls(config)
        """初始化工具连接"""
        try:
            logger.info(f"Connecting to MCP server at {self.config.MCP_SERVER_URL}")
            self.client = MultiServerMCPClient({
                "kubernetes": {
                "url": self.config.MCP_SERVER_URL,  # 请确保服务启动并监听这个 URL
                "transport": "sse"
                }
            })
            
            self.tools = await self.client.get_tools()
            logger.info(f"Successfully loaded {len(self.tools)} tools")
            
            # 打印可用工具
            for tool in self.tools:
                logger.debug(f"Available tool: {tool.name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")
            raise
        return self
    
    def get_tool(self):
        """获取工具列表"""
        return self.tools
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            # 这里可以添加具体的健康检查逻辑
            return len(self.tools) > 0
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

class KubernetesChatBot:
    """Kubernetes聊天机器人"""
    
    def __init__(self, config: Config, tool_manager, llm, llm_with_tools, graph):
        self.config = config
        self.tool_manager = tool_manager
        self.llm = llm  # 提供访问原始 LLM
        self.llm_with_tools = llm_with_tools
        self.graph = graph

    @classmethod
    async def create(cls, config: Config):
        """异步构造函数"""
        # 初始化工具
        tool_manager = await KubernetesToolManager.create(config)

        # 初始化 LLM
        llm = ChatDeepSeek(
            model=config.DEEPSEEK_MODEL,
            api_key=config.DEEPSEEK_API_KEY,
            #timeout=config.REQUEST_TIMEOUT
        )
        logger.info("LLM initialized successfully")

        tools = tool_manager.get_tool()

        # 包装工具
        llm_with_tools = llm.bind_tools(tools)
        def build_chatbot_node(llm_with_tools, config):
            async def chatbot_node(state: State) -> Dict[str, Any]:
                retry_count = 0
                last_exception = None

                while retry_count < config.MAX_RETRIES:
                    try:
                        messages = [HumanMessage(content=m.content) for m in state["messages"]]

                        logger.info(f"Processing request (attempt {retry_count + 1})")
                        start_time = time.time()

                        response = await llm_with_tools.ainvoke(messages)

                        duration = time.time() - start_time
                        logger.info(f"Request processed in {duration:.2f}s")

                        return {
                            "messages": [response],
                            "last_tool_call": None  # 如果你有提取工具调用可以加上
                        }

                    except Exception as e:
                        retry_count += 1
                        last_exception = e
                        logger.error(f"Chatbot error (attempt {retry_count}): {e}")

                        if retry_count < config.MAX_RETRIES:
                            await asyncio.sleep(2 ** retry_count)

                return {
                    "messages": [AIMessage(content="抱歉，系统暂时无法处理您的请求，请稍后重试。")]
                }

            return chatbot_node
        chatbot_node = build_chatbot_node(llm_with_tools, config)

        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", chatbot_node)
        graph_builder.add_node("tools", ToolNode(tools=tools))
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")
        memory = MemorySaver()
        graph = graph_builder.compile(checkpointer=memory)

        return cls(config, tool_manager, llm, llm_with_tools, graph)
    
    
    def _prepare_messages(self, messages: list) -> list:
        """准备消息，添加系统提示"""
        system_prompt = """你是一个专业的Kubernetes管理助手。你的任务是帮助用户管理和查询Kubernetes集群。

请遵循以下原则：
1. 优先使用提供的工具来获取实时的集群信息
2. 提供准确、详细的信息和建议
3. 如果操作可能有风险，请明确警告用户
4. 使用清晰、易懂的语言回答
5. 如果不确定，请说明不确定性

当前可用的工具可以帮你：
- 查询节点状态和资源使用情况
- 检查Pod调度状态
- 分析集群健康状况
- 提供运维建议
"""
        
        # 如果没有系统消息，添加一个
        has_system_msg = any(isinstance(msg, SystemMessage) for msg in messages)
        if not has_system_msg:
            messages = [SystemMessage(content=system_prompt)] + messages
        
        return messages
    
    def _extract_tool_call(self, response) -> Optional[str]:
        """提取工具调用信息"""
        if hasattr(response, 'tool_calls') and response.tool_calls:
            return response.tool_calls[0].get('name', 'unknown')
        return None
    

    async def chat(self, user_input: str, thread_id: str = "default") -> None:
        """执行对话"""
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            logger.info(f"Starting chat session: {thread_id}")
            
            async for event in self.graph.astream(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "user_context": {"timestamp": time.time()}
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
        
        except Exception as e:
            logger.error(f"Chat execution failed: {e}")
            print(f"聊天执行失败: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """系统健康检查"""
        status = {
            "llm_status": "unknown",
            "tools_status": "unknown",
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
        
        # 总体状态
        if status["llm_status"] == "healthy" and status["tools_status"] == "healthy":
            status["overall_status"] = "healthy"
        else:
            status["overall_status"] = "unhealthy"
        
        return status

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
        user_input = "请帮我查询 node-1 的调度状态，并判断是否可以分配 Pod abc-123。"
        
        print("=== Kubernetes ChatBot ===")
        print(f"用户输入: {user_input}")
        print("\n机器人回复:")
        
        await chatbot.chat(user_input, "kube-session-1")
        
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        print(f"应用启动失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())
