import streamlit as st
import requests
import json

IPADDRESS = "localhost"
AGENT_PORT = "8081"
AGENT_BASE_URL = f'http://{IPADDRESS}:{AGENT_PORT}'

def create_session():
    url = AGENT_BASE_URL + "/create_session"
    headers = {
        "accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        try:
            data = response.json()
            session_id = data.get("session_id")
            return session_id
        except ValueError:
            st.error("Failed to parse session ID.")
            return None
    else:
        st.error(f"Failed to create session. Status Code: {response.status_code}")
        return None

def generate_response(user_id, session_id, user_message):
    url = AGENT_BASE_URL + "/generate"
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json"
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": user_message
            }
        ],
        "user_id": user_id,
        "session_id": session_id
    }

    response = requests.post(url, json=payload, headers=headers)
    return response

def parse_response(response_text):
    result = ""
    try:
        for line in response_text.splitlines():
            if line.startswith("data:"):
                json_data = json.loads(line.replace("data: ", ""))
                content = json_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                result += content
    except json.JSONDecodeError:
        st.error("Error parsing response data.")
    return result

def main():
    if "user_id" not in st.session_state:
        st.session_state.user_id = "4"
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    st.session_state.user_id = st.selectbox("Select Student ID:", options=[str(i) for i in range(1, 11)], index=int(st.session_state.user_id) - 1)

    if st.button("Create New Session"):
        session_id = create_session()
        if session_id:
            st.session_state.session_id = session_id
            st.success(f"New session created: {session_id}")

    if st.session_state.session_id:
        st.markdown(f"**Current Session ID:** {st.session_state.session_id}")

    example_prompts = [
        "What is my physics course grade distribution?",
        "What are all of my exam scores?",
        "What did I get on homework 2 and 3 in thermo?",
        "What do discussions say about relativity?"
    ]

    user_message = st.selectbox("Choose a query or type your own:", options=["Type your own..."] + example_prompts, index=0)

    if user_message == "Type your own...":
        user_message = st.text_input("Enter your query:", placeholder="Ask something...")

    if st.button("Send") and user_message and st.session_state.session_id:
        with st.spinner("Processing your request..."):
            response = generate_response(st.session_state.user_id, st.session_state.session_id, user_message)
            
            st.subheader("Agent Response")

            if response.status_code == 200:
                if "data:" in response.text:
                    parsed_output = parse_response(response.text)
                    st.markdown(parsed_output)
                else:
                    try:
                        response_data = response.json()
                        assistant_message = response_data.get("choices", [{}])[0].get("message", {}).get("content", "No content available.")
                        st.markdown(assistant_message)
                    except ValueError:
                        st.error("Unexpected response format. Please try again later.")
                        st.text(response.text)
            else:
                st.error(f"Error: {response.status_code}")
                st.text(response.text)

if __name__ == "__main__":
    main()
