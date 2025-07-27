from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_deepseek import ChatDeepSeek


client = MultiServerMCPClient(
    {
        "kubernetes": {
            "url": "http://192.168.100.99:30088",  # MCP server 地址
            "transport": "streamable_http",   # SSE 模式
        }
    }
)
tools =  client.get_tools()
# 可替换为你自己的 LLM，例如 deepseek-chat 本地地址
agent = create_react_agent(
    llm=ChatDeepSeek(model="deepseek-chat", api_key="sk-2"),
    tools=tools
)
# 请求示例
response = agent.ainvoke({"messages": "List all pods in the kube-system namespace"})
print(response)

