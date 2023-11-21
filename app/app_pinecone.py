import os

import chainlit as cl
import pinecone
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # type: ignore
    environment=PINECONE_ENV,  # type: ignore
)


def set_custom_prompt(prompt_template):
    
    """
    Prompt template for QA retrieval for each vectorstore
    """
    temp = cl.user_session.get("document")
    prompt = PromptTemplate(
        template=prompt_template , input_variables=["context", "question"]
    )
    
    return prompt


def create_retrieval_qa_chain(llm, prompt, db):
    """
    Creates a Retrieval Question-Answering (QA) chain using a given language model, prompt, and database.

    This function initializes a RetrievalQA object with a specific chain type and configurations,
    and returns this QA chain. The retriever is set up to return the top 3 results (k=3).

    Args:
        llm (any): The language model to be used in the RetrievalQA.
        prompt (str): The prompt to be used in the chain type.
        db (any): The database to be used as the retriever.

    Returns:SS
        RetrievalQA: The initialized QA chain.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr",search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


def load_model():
    """
    Load a locally downloaded model.

    Parameters:
        model_path (str): The path to the model to be loaded.
        model_type (str): The type of the model.
        max_new_tokens (int): The maximum number of new tokens for the model.
        temperature (float): The temperature parameter for the model.

    Returns:
        CTransformers: The loaded model.

    Raises:
        FileNotFoundError: If the model file does not exist.
        SomeOtherException: If the model file is corrupt.
    """
    # model_name = "text-davinci-003"
    model_name = "gpt-3.5-turbo"
    # model_name = "gpt-4"
    llm = ChatOpenAI(model_name=model_name, temperature=1)

    return llm

def create_retrieval_qa_bot(
    prompt_template,
    model_name="ada",
    index_name=PINECONE_INDEX_NAME,
    device="cuda",
    
):
    
    """
    This function creates a retrieval-based question-answering bot.

    Parameters:
        model_name (str): The name of the model to be used for embeddings.
        persist_dir (str): The directory to persist the database.
        device (str): The device to run the model on (e.g., 'cpu', 'cuda').

    Returns:
        RetrievalQA: The retrieval-based question-answering bot.

    Raises:
        FileNotFoundError: If the persist directory does not exist.
        SomeOtherException: If there is an issue with loading the embeddings or the model.
    """

    try:
        embeddings = OpenAIEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        )
    except Exception as e:
        raise Exception(
            f"Failed to load embeddings with model name {model_name}: {str(e)}"
        )

    db = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

    try:
        llm = load_model()  # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

    qa_prompt = (
        set_custom_prompt(prompt_template)
    )  # Assuming this function exists and works as expected

    try:
        qa = create_retrieval_qa_chain(
            llm=llm, prompt=qa_prompt, db=db
        )  # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to create retrieval QA chain: {str(e)}")

    return qa

@cl.on_chat_start
async def initialize_bot():
    """
    Initializes the bot when a new chat starts.

    This asynchronous function creates a new instance of the retrieval QA bot,
    sends a welcome message, and stores the bot instance in the user's session.
    """
    
    welcome_message = cl.Message(content="Starting the bot...",)
    await welcome_message.send()
    welcome_message.content = (
        "Hi, Can I help you?."
    )
    await welcome_message.update()
    
    prompt_template = """Sử dụng các phần ngữ cảnh sau đây để trả lời câu hỏi của người dùng.
    
    Phần "NGUỒN THAM KHẢO" nên là một tham khảo đến nguồn tài liệu mà bạn đã lấy câu trả lời của mình.
    Ví dụ về câu trả lời của bạn nên được viết như sau:
    Ngữ cảnh: {context}
    Câu hỏi: {question}

    """
    # +"CV của tôi:"
    # + documents+"""
    # CV này là của tôi. Hãy dùng nó và thực hiện những câu hỏi.
    # Chỉ trả về phần câu trả lời hữu ích bên dưới, không bao gồm bất kỳ điều gì khác.
    # Câu trả lời hữu ích:
    # """
    qa_chain = create_retrieval_qa_bot(prompt_template=prompt_template)
    cl.user_session.set("chain", qa_chain)

@cl.on_message
async def process_chat_message(message):
    """
    Processes incoming chat messages.

    This asynchronous function retrieves the QA bot instance from the user's session,
    sets up a callback handler for the bot's response, and executes the bot's
    call method with the given message and callback. The bot's answer and source
    documents are then extracted from the response.
    """
    
    qa_chain = cl.user_session.get("chain")
    callback_handler = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    
    callback_handler.answer_reached = True
    response = await qa_chain.acall(message, callbacks=[callback_handler])
    bot_answer = response["result"]
    # source_documents = response["source_documents"]

    # if source_documents:
    #     bot_answer += f"\nSources:" + str(source_documents)
    # else:
    #     bot_answer += "\nNo sources found"

    await cl.Message(content=bot_answer).send() 