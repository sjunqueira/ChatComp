import os
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools

from langchain.agents.initialize import initialize_agent


llm = ChatOpenAI(model="gpt-3.5-turbo")

tools = load_tools(['ddg-search', 'wikipedia'], llm=llm)

#print(tools[1].name, tools[1].description)

agent = initialize_agent(
    tools,
    llm,
    verbose = True
)

#print(agent.agent.llm_chain.prompt.template)

query = """
Vou viajar para o Canadá em Outubro de 2024, quero um cronograma completo, de custos de passagens de avião de Florianópolis até Vancouver até passeios e eventos que irão ocorrer em Outubro e seus custos. A viagem será feita em duas pessoas adultas casadas.
"""


agent.run(query)