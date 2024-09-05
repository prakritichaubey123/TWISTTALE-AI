# Importing Dependencies
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Faiss Index Path
FAISS_INDEX = r"C:/Users/Prakriti Chaubey/Desktop/kiku/AI/Python/projects/TWISTTALE-AI/vectorstore"

# Custom prompt template (updated)
custom_prompt_template = """[INST] <<SYS>>
You are a master storyteller and plot twist genius, having absorbed the essence of countless novels and stories. 
Your goal is to analyze the following scene from a novel, capturing its atmosphere, characters, and underlying tension. 
If the input does not make sense or is not factually coherent, explain why instead of generating something incorrect. If you don't have enough information to create a meaningful twist, don't provide false information.
Do not say thank you or mention that you are an AI Assistant. Be open and direct in your response.
<</SYS>>
Use the following pieces of context to craft a unique and shocking plot twist that seamlessly fits into the narrative.
Context: {context}
Scene Description: {query}
Plot Twist: [/INST]
"""

# Return the custom prompt template
def set_custom_prompt_template():
    """
    Set the custom prompt template for the LLMChain
    """
    print("Setting custom prompt template...")
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "query"])
    print("Custom prompt template set successfully.")
    return prompt

# Return the LLM
def load_llm():
    """
    Load the LLM entirely on CPU without using device map.
    """
    # Model ID
    repo_id = "SweatyCrayfish/llama-3-8b-quantized"

    # Load the model on CPU without using device_map or accelerate
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype="auto"  # Automatically select the appropriate dtype (e.g., float32 on CPU)
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        use_fast=True
    )

    # Create the pipeline for text generation (without the device argument)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512
    )

    # Return the HuggingFace pipeline
    return HuggingFacePipeline(pipeline=pipe)

# Create the Retrieval QA chain
def retrieval_qa_chain(llm, prompt, db):
    """
    Create the Retrieval QA chain
    """
    # Create the chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

    return qa_chain

# Return the QA pipeline
def qa_pipeline():
    """
    Create the QA pipeline
    """
    print("Creating the QA pipeline...")
    try:
        # Load the HuggingFace embeddings
        print("Loading HuggingFace embeddings...")
        embeddings = HuggingFaceEmbeddings()
        print("Embeddings loaded successfully.")

        # Load the FAISS index
        print(f"Loading FAISS index from {FAISS_INDEX}...")
        db = FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)  # Set to True
        print("FAISS index loaded successfully.")

        # Load the LLM
        llm = load_llm()

        # Set the custom prompt template
        qa_prompt = set_custom_prompt_template()

        # Create the retrieval QA chain
        chain = retrieval_qa_chain(llm, qa_prompt, db)

        print("QA pipeline created successfully.")
        return chain

    except Exception as e:
        print(f"An error occurred while creating the QA pipeline: {e}")
        raise

if __name__ == "__main__":
    print("Running utils.py...")
    qa_pipeline()
    print("Finished running utils.py.")
#giugekbhjbdchs