import streamlit as st
from faq import process_folder, generate_answer, vector_store  # import vector_store to check init
from sql import sql_chain
from pathlib import Path
from router import router
from PIL import Image
import base64


# -------------------------------
# Initialize placement data
# -------------------------------
if "initialized" not in st.session_state:
    folder = "resources/placement_texts"
    for msg in process_folder(folder, reset=False):
        print(msg)
    st.session_state["initialized"] = True

    
# -------------------------------
# Ask function
# -------------------------------
def ask(query):
    route = router(query).name
    if route == 'faq':
        answer, context_s = generate_answer(query)   # returns (answer, sources)
        print("faq route")
        #print(context_s[:3])
        return answer

    elif route == 'sql':
        print("sql route")
        return sql_chain(query)

    else:
        print("unwanted question")
        return '''I'm designed to answer queries about placements and related details .Please try asking about training, vision & mission, faculty, or hiring statistics.'''

# -------------------------------
# UI
# -------------------------------
st.markdown("<h1 style='text-align: center;'>Welcome to Placement Bot</h1>", unsafe_allow_html=True)

# Add and center college logo
logo_path = Path(__file__).parent / "resources/bnmit_logo.png"
logo = Image.open(logo_path)

def image_to_base64(img):
    from io import BytesIO
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

logo_base64 = image_to_base64(logo)
st.markdown(
    f'<div style="text-align: center;"><img src="data:image/png;base64,{logo_base64}" width="200"></div>',
    unsafe_allow_html=True
)

# Chat interface
query = st.chat_input("Ask anything about placements...")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Handle new query
if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    response = ask(query)

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
