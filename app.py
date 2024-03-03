import openai
from environs import Env
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.document_loaders import CSVLoader

# Libraries
# pip install openai
# pip install langchain
# pip install unstructured
# pip install tiktoken
# pip install chromadb
# pip install langchain-community

# env = Env()
# env.read_env()
# sys.path.append('../..')
openai.api_key = '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'
loader = CSVLoader("final.csv")
pages = loader.load()

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(pages)

embedding = OpenAIEmbeddings()
llm_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=llm_name, temperature=0)
# llm.predict("Hello world!")
# persist_directory = 'doc/chroma/'
# vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
print("Beginning")
vectordb = Chroma.from_documents(pages, embedding)
print('Database Loaded')

# Build prompt
template_uz = """Use the following pieces of context to answer the question at the end.
Please write all of the information in uzbek language. If the user asks to find a disease by symptoms, firstly, give three possible diseases.
Then you have to include precaution for all three possible diseases. The last thing you have to do is include Description for only first possible disease.

{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT_uz = PromptTemplate(input_variables=["context", "question"], template=template_uz)

# Run chain
qa_chain_uz = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever(), return_source_documents=False,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT_uz})

def yes_man(question, history):
   return str(qa_chain_uz({"query": question})['result'])

import gradio as gr

chat_interface = gr.ChatInterface(
    yes_man,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Type", container=False, scale=7),
    title="Doctor Consultant",
    description="!!! Bu shunchaki suniy intelekt tomonidan berilgan maslahat ko'proq ma'lumot olish uchun tajribali doktorlar bilan bog'laning !!!",
    theme="soft",
    examples=["Menda ko'zlarning sarg'ayishi, mushak og'rig'i, terining sarg'ayishi va qorin og'rig'i kuzatilyapti"],
    retry_btn=None,
    undo_btn="O'chirish",
    clear_btn="Tozalash",
)
# chat_interface.launch(share=False)
import torch
import gradio as gr
from fastai import *
from fastai.vision.all import *

# Load the trained model
learn = load_learner("model.pkl")

# Specify the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

text_input = gr.Image()

def image_processing_function(image):
    # Convert Gradio image object to a PyTorch tensor
    image_tensor = torch.tensor(image)

    # Make predictions using your model
    predictions = learn.predict(image_tensor, with_input=True)

    # Return predictions (assuming it's a text output)
    return f"{predictions[1]}: {int(predictions[3][predictions[3].argmax()]*100)}%"

pneumo_interface = gr.Interface(
    fn=image_processing_function,
    inputs=text_input,
    outputs="text",
    title="Pnevmaniya",
    description="""!!! Bu shunchaki suniy intelekt tomonidan berilgan maslahat. Ko'proq ma'lumot olish uchun tajribali doktorlar bilan bog'laning !!! 
    Iltimos faqat o'pkaga oid bo'lgan sifatli rasm kiriting!""",
    theme="soft",
    examples=["data/norm.jpg"],
    live=False
)

# pneumo_interface.launch(share=False)
with gr.Blocks() as consult_interface:
    with gr.Column():
        with gr.Row():
            btn1 = gr.Button("Xususiy Shifoxonalar", link='')
            btn2 = gr.Button("Davlat Shifoxonalari", link='')
# consult_interface.launch()

final = gr.TabbedInterface([chat_interface, pneumo_interface, consult_interface], tab_names=['Chatbot', 'Rasm Tashxisi', 'Konsultatsiya'])
final.launch(share=True)