import streamlit as st
from utils import qa_pipeline

# Load the QA pipeline
chain = qa_pipeline()
#gbsdkdhbajs
def main():
    # Set the title of the web application
    st.title('Twist Tale AI')

    # Initialize the session state if it doesn't exist
    if 'chat_log' not in st.session_state:
        st.session_state.chat_log = []

    # Get the user's question or scene description
    user_input = st.text_area("Enter your scene description here:")

    # Process user input and generate a response
    if st.button('Generate Plot Twist') and user_input:
        # Prepare the input for the chain (context is empty in this case)
        input_data = {
            "query": user_input,  # The scene description provided by the user
            "context": ""  # Context can be modified or left empty depending on the use case
        }

        # Generate the answer using the QA pipeline
        try:
            response = chain(input_data)  # Calling the chain with the input
            bot_output = response['result']  # Extract the plot twist from the response
            source_documents = response['source_documents']  # Extract the supporting documents, if any

            # Add the user input and bot output to the chat log
            st.session_state.chat_log.append({"User": user_input, "Bot": bot_output})

            # Optionally display the source documents (if you want to show them)
            if source_documents:
                st.write("Supporting Documents:")
                for doc in source_documents:
                    st.write(doc.page_content)  # Display the content of the supporting document

        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Display the chat log (history of scene inputs and plot twists)
    for exchange in st.session_state.chat_log:
        st.markdown(f'**Scene Description:** {exchange["User"]}')
        st.markdown(f'**Plot Twist:** {exchange["Bot"]}')

if __name__ == "__main__":
    main()
