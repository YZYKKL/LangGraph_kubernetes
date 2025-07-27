import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_deepseek import ChatDeepSeek
async def main():
    # 初始化 MCP 多服务客户端
    client = MultiServerMCPClient({
        "kubernetes": {
            "url": "http://192.168.100.99:30088/sse",  # 请确保服务启动并监听这个 URL
            "transport": "sse"
        }
    })

    # 异步获取工具
    try:
        tools = await client.get_tools()
    except Exception as e:
        print("❌ MCP 工具加载失败:", e)
    
    # 初始化 DeepSeek 模型
    llm = ChatDeepSeek(model="deepseek-chat", api_key="sk-")

    # 创建 LangGraph agent
    agent = create_react_agent(llm, tools)

    # 发送请求
    response = await agent.ainvoke({"messages": "List all pods in the kube-system namespace"})
    print(response)

# 执行 async 主函数
if __name__ == "__main__":
    asyncio.run(main())
