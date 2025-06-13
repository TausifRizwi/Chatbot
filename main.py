# main.py (Streamlit interface)

import streamlit as st
from subject_agent import SubjectAgent

def initialize_agents():
    """
    Initializes and returns a dictionary of SubjectAgent instances.
    This function uses Streamlit's @st.cache_resource decorator to ensure
    agents are initialized only once across app reruns.
    """
    # Use st.cache_resource to avoid re-initializing agents on every rerun
    # This is crucial for performance in Streamlit apps.
    if 'polity_agent' not in st.session_state:
        st.session_state.polity_agent = SubjectAgent("Polity")
    if 'history_agent' not in st.session_state:
        st.session_state.history_agent = SubjectAgent("History")
    if 'economics_agent' not in st.session_state:
        st.session_state.economics_agent = SubjectAgent("Economics")

    return {
        "Polity": st.session_state.polity_agent,
        "History": st.session_state.history_agent,
        "Economics": st.session_state.economics_agent
    }

def main():
    """
    Main function to run the multi-subject Q&A agent system with Streamlit.
    """
    st.set_page_config(page_title="Advanced Subject Q&A Agent", layout="centered")

    st.title("📚 Advanced Multi-Subject Q&A System")
    st.markdown("Ask questions across different subjects! (Using internal knowledge bases)")

    # Initialize agents (or retrieve them from cache)
    agents = initialize_agents()

    # Sidebar for subject selection
    st.sidebar.header("Select a Subject")
    subject_names = list(agents.keys())
    selected_subject = st.sidebar.radio("Choose a subject:", subject_names, key="subject_radio")

    # Main content area for questions and answers
    st.header(f"Question & Answer for {selected_subject}")

    # Text input for the question
    question = st.text_input(f"Enter your question about {selected_subject}:", key="question_input")

    # Button to submit the question
    if st.button("Get Answer", key="get_answer_button"):
        if question:
            # Get the selected agent
            current_agent = agents[selected_subject]

            # Get the answer
            answer = current_agent.answer_question(question)

            # Display the answer
            st.subheader("Answer:")
            st.info(answer) # Using st.info for a nice blue box display
        else:
            st.warning("Please enter a question to get an answer.")

    st.markdown("---")
    st.markdown(
        """
        <small>This system combines exact Q&A pairs with keyword search within larger,
        pre-defined textual content. It does not use external AI APIs for generation.</small>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()