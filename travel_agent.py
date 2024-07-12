import os
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Definindo o modelo e as ferramentas
model = ChatOpenAI(model="gpt-3.5-turbo")
tools = load_tools(['ddg-search', 'wikipedia'], llm=model)
prompt = hub.pull("hwchase17/react")

# Definindo a consulta
query = """
Vou viajar para o Canadá em Outubro de 2024, quero um cronograma completo, de custos de passagens de avião de Florianópolis até Vancouver até passeios e eventos que irão ocorrer em Outubro e seus custos. A viagem será feita em duas pessoas adultas casadas.
"""
# Criando o agente
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Executando a consulta
agent_executor.invoke({"input": query})
