import streamlit as st
import os
from dotenv import load_dotenv
from main import MedicalQuerySystem

# Load environment variables at startup
if os.path.exists(".env"):
    load_dotenv()

# Use Streamlit secrets if available
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def initialize_chat():
    try:
        if "medical_system" not in st.session_state:
            # Verify API key is available and valid
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OpenAI API key not found. Please check your .env file or Streamlit secrets.")
                return
            if not (api_key.startswith("sk-") or api_key.startswith("sk-proj-")):
                st.error("Invalid OpenAI API key format. Key should start with 'sk-' or 'sk-proj-'")
                return
                
            st.session_state.medical_system = MedicalQuerySystem(debug=False)
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_mode" not in st.session_state:
            st.session_state.current_mode = "General"
        if "user_initialized" not in st.session_state:
            st.session_state.user_initialized = False
    except Exception as e:
        st.error(f"Error initializing chat: {str(e)}")
        return

def handle_mode_change():
    new_mode = st.session_state.mode_selector.lower()
    system_message = None
    
    if new_mode == "general":
        st.session_state.medical_system.switch_category("switch gen")
        # Clear any mode-specific messages
        if "mode_info" in st.session_state:
            del st.session_state.mode_info
    elif new_mode == "iols":
        st.session_state.medical_system.switch_category("switch iols")
        system_message = (
            "In IOL mode, I can provide detailed information about the Precizon Presbyopic NVA IOL. "
            "This is a premium intraocular lens designed for presbyopia correction. "
            "Please feel free to ask any questions about the Precizon Presbyopic NVA IOL's features, "
            "specifications, or clinical applications."
        )
        st.session_state.mode_info = system_message
    else:  # CTR mode
        st.session_state.medical_system.switch_category("switch ctr")
        system_message = (
            "In CTR mode, I can provide information about the following Capsular Tension Ring models:\n"
            "1. RingJect Model 376\n"
            "2. RingJect Model 375\n"
            "3. CTR Model 275 12/10\n"
            "4. CTR Model 276 13/11\n\n"
            "These CTR models are designed for capsular support during cataract surgery. "
            "Please feel free to ask about their specifications, indications, or surgical techniques."
        )
        st.session_state.mode_info = system_message
    
    # Add system message to both Streamlit and medical system chat histories
    if system_message:
        # Add to Streamlit messages
        st.session_state.messages.append({"role": "assistant", "content": system_message})
        
        # Add to medical system chat history for the current category
        current_history = st.session_state.medical_system.get_current_history()
        current_history.append({"role": "assistant", "content": system_message})
        
    st.session_state.current_mode = st.session_state.mode_selector

def handle_start_chat():
    if st.session_state.name and st.session_state.role:
        st.session_state.user_initialized = True
        # Format name with title if needed
        display_name = st.session_state.name
        if st.session_state.role == "Ophthalmologist" and not display_name.lower().startswith("dr"):
            display_name = f"Dr. {display_name}"
            
        greeting = (
            f"Welcome {display_name}! I am Sam, your dedicated educational assistant from Ophtec. "
            "I specialize in providing comprehensive information about Ophtec's IOLs and CTRs, "
            "and I'm also well-versed in general ophthalmology topics. "
            "Please select your preferred mode from the dropdown menu below to begin our conversation."
        )
        st.session_state.messages = [{"role": "assistant", "content": greeting}]

def main():
    st.set_page_config(
        page_title="Ophtec's Knowledge Bot",
        layout="wide"
    )
    
    # Simple CSS for basic styling
    st.markdown("""
        <style>
        .title {
            color: white;
            font-size: 2rem;
            font-weight: 600;
            margin-bottom: 2rem;
        }
        
        /* Style the bottom mode selector */
        div.stSelectbox[data-testid="stSelectbox"]:not([data-testid="stSidebar"] *) {
            max-width: 150px !important;
            margin: 0 auto;
            position: fixed;
            bottom: 10px;
            right: 80px;
            transform: none;
            z-index: 1000;
            padding-bottom: 10px;
            background: #0E1117;
        }

        /* Add mode selector label */
        .mode-label {
            position: fixed;
            bottom: 25px;
            right: 240px;
            color: white;
            z-index: 1000;
            font-size: 16px;
            background: #0E1117;
            padding: 5px 10px;
            display: flex;
            align-items: center;
        }

        /* Style the mode selector dropdown */
        div.stSelectbox[data-testid="stSelectbox"]:not([data-testid="stSidebar"] *) > div > div {
            background-color: #FF4B4B !important;
            color: white !important;
        }

        /* Style the dropdown options */
        div.stSelectbox[data-testid="stSelectbox"] div[role="listbox"] div {
            background-color: #FF4B4B !important;
            color: white !important;
        }

        /* Style the dropdown when expanded */
        div.stSelectbox[data-testid="stSelectbox"] div[role="listbox"] {
            background-color: #FF4B4B !important;
        }

        /* Style sidebar selectbox */
        [data-testid="stSidebar"] [data-testid="stSelectbox"] {
            margin-top: 1rem !important;
            margin-bottom: 1rem !important;
            background: #262730;
        }

        [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
            background: #262730;
        }
        
        /* Hide the labels */
        div[data-testid="stSelectbox"] > label {
            display: none;
        }

        /* Adjust chat input position */
        .stChatInput {
            padding-bottom: 40px !important;
        }

        /* Style sidebar elements */
        .sidebar-content {
            padding: 1rem;
        }

        /* Remove padding from sidebar */
        section[data-testid="stSidebar"] {
            padding-top: 0rem;
        }

        /* Style the Welcome text */
        .welcome-text {
            font-size: 24px;
            color: white;
            margin-bottom: 2rem;
        }

        /* Style the Start Chat button */
        .stButton > button {
            background-color: #FF4B4B !important;
            color: white !important;
            border: none !important;
            border-radius: 4px !important;
            width: 100% !important;
            margin-top: 1rem !important;
        }

        /* Style text input */
        .stTextInput input {
            background-color: #262730 !important;
            color: white !important;
            border: none !important;
            padding: 8px 12px !important;
        }

        /* Add spacing between elements */
        .element-container {
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize chat state
    initialize_chat()

    # Sidebar for user initialization
    with st.sidebar:
        st.markdown('<p class="welcome-text">Welcome!</p>', unsafe_allow_html=True)
        
        # Create columns for better spacing
        col1, col2, col3 = st.columns([1, 10, 1])
        
        with col2:
            # Name input
            name = st.text_input("Name", key="name", label_visibility="collapsed", placeholder="Name")
            
            # Role selection
            role = st.selectbox(
                "Select your role",
                ["Sales Rep", "Ophthalmologist"],
                key="role",
                label_visibility="collapsed"
            )
            
            # Start Chat button
            st.button(
                "Start Chat",
                key="start_chat",
                on_click=handle_start_chat,
                use_container_width=True
            )

    # Main chat interface
    if st.session_state.user_initialized:
        # Title
        st.markdown("<h1 class='title'>Ophtec's Knowledge Bot</h1>", unsafe_allow_html=True)
        
        # Display mode-specific information if available
        if "mode_info" in st.session_state:
            st.info(st.session_state.mode_info)
        
        # Chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your query here"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.medical_system.process_query(prompt)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

        # Mode selector - small and centered
        st.markdown('<div class="mode-label">Select mode: </div>', unsafe_allow_html=True)
        st.selectbox(
            "Select Mode",
            ["General", "IOLs", "CTR"],
            key="mode_selector",
            on_change=handle_mode_change,
            index=["General", "IOLs", "CTR"].index(st.session_state.current_mode),
            label_visibility="collapsed"  # Hide the label
        )

if __name__ == "__main__":
    main() 