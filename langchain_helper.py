from dotenv import load_dotenv
import os
from langchain_openai import OpenAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from few_shots import few_shots


# Calling api key from .env
load_dotenv()
apikey = os.getenv('OPENAI_API_KEY')

def get_few_shots_db_chain():
    llm = OpenAI(temperature=0.7)

    #Database info
    db_user = "test"
    db_password = os.getenv("db_password")
    db_host = "localhost"
    db_name="atliq_tshirts"

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}", sample_rows_in_table_info=3)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    to_vectorize = [" ".join(example.values()) for example in few_shots]
    vectorstore = Chroma.from_texts(to_vectorize, embedding=embeddings, metadatas= few_shots)
    example_selector = SemanticSimilarityExampleSelector(vectorstore=vectorstore, k=2)

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(example_selector=example_selector,
                        example_prompt=example_prompt,
                        prefix=_mysql_prompt,
                        suffix=PROMPT_SUFFIX,
                        input_variables=["input", "table_info", "top_k"])

    chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)

    return chain

if __name__=="__main__":
    chain = get_few_shots_db_chain()
    print(chain.run("If we have to sell all the Levi’s T-shirts today. How much revenue our store will generate without discount?"))



