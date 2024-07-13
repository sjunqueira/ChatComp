import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
#Imports para a função load_data
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
import bs4

# Definindo o modelo e as ferramentas
model = ChatOpenAI(model="gpt-3.5-turbo")

query = """
Quero viajar para Londres em Outubro de 2024, encontre preços de passagens e eventos que irão acontecer em Outubro, quero um cronograma completo, de custos de passagens de avião de Florianópolis até Londres até custos de passeios e eventos que irão acontecer. 
A viagem será feita em duas pessoas adultas casadas.
"""

def researchAgent(model, query):
    tools = load_tools(['ddg-search', 'wikipedia'], llm=model)
    prompt = hub.pull("hwchase17/react")
    # Criando o agente
    agent = create_react_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, prompt=prompt, verbose=True)
    web_context = agent_executor.invoke({"input": query})
    return web_context['output']
    
print(researchAgent(model, query))

#Fazer toda a parte verde do fluxograma
def loadData():
    #Carregar a página
    loader = WebBaseLoader(
    web_path=("https://www.dicasdeviagem.com/inglaterra/"),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("postcontentwrap","pagetitleloading background-imaged loading-dark"))))
    docs = loader.load()
    #Separar o conteúdo da página em partes menores (Chunk_size é quantos Tokens cada parte vai ter, e Chuck overlap é a sobreposição que vai ter entre esses chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    return retriever

def getRelevantDocs(query):
    retriever = loadData()
    relevant_docs = retriever.invoke(query)
    print(relevant_docs)
    return relevant_docs

def supervisorAgent(model, query, web_context, relevant_docs):
    prompt_template = """
    Você é um gerente de uma agência de viagens, sua resposta final deverá ser um roteiro de viagem completo e detalhado. 
    Utilize o contexto de eventos e preços de passagens, o input do usuário e também os documentos relevantes para elaborar o roteiro.
    Contexto: {web_context}
    Documentos relevantes: {relevant_docs}
    Input Usuário: {query}
    Assistente: 
    """

    prompt = PromptTemplate(
        input_variables=['web_context','relevant_docs', 'query'],
        template=prompt_template
    )

    sequence = RunnableSequence(prompt | model)

    response = sequence.invoke({'web_context': web_context,'relevant_docs': relevant_docs, 'query': query})
    return response

def getResponse(query, model):
    web_context = researchAgent(model, query)
    relevant_docs = getRelevantDocs(query)
    response = supervisorAgent(model, query, web_context, relevant_docs)

    return response

print(getResponse(query, model).content)
