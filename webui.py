import os
import subprocess

import streamlit as st

from fuzzy.enums import EnvironmentVariables
from fuzzy.handlers.attacks.base import attack_handler_fm
from fuzzy.handlers.attacks.enums import FuzzerAttackMode
from fuzzy.handlers.classifiers.base import classifiers_fm
from fuzzy.handlers.classifiers.enums import Classifier
from fuzzy.llm.providers.base import llm_provider_fm
from fuzzy.llm.providers.enums import LLMProvider

st.set_page_config(
    page_title="FuzzyAI Web UI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.image("resources/logo.png", width=175)

if 'env_vars' not in st.session_state:
    st.session_state.env_vars = {}
if 'verbose' not in st.session_state:
    st.session_state.verbose = False
if 'db_address' not in st.session_state:
    st.session_state.db_address = "127.0.0.1"
if 'max_workers' not in st.session_state:
    st.session_state.max_workers = 1
if 'max_tokens' not in st.session_state:
    st.session_state.max_tokens = 100
if 'extra_params' not in st.session_state:
    st.session_state.extra_params = {}
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = []
if 'selected_attacks' not in st.session_state:
    st.session_state.selected_attacks = []
if 'selected_classifiers' not in st.session_state:
    st.session_state.selected_classifiers = []
    
st.sidebar.header("Environment Settings")
api_keys = [x.value for x in EnvironmentVariables]
new_env_key = st.sidebar.selectbox("Name", options=api_keys)
new_env_value = st.sidebar.text_input("Value")
if st.sidebar.button("Add Variable"):
    if new_env_key and new_env_value:
        st.session_state.env_vars[new_env_key] = new_env_value
for key, value in st.session_state.env_vars.items():
    st.sidebar.write(f"{key}: {value}")
st.session_state.verbose = st.sidebar.checkbox("Verbose Logging", value=st.session_state.verbose)
st.session_state.db_address = st.sidebar.text_input("MongoDB Address", value=st.session_state.db_address)
st.session_state.max_workers = st.sidebar.number_input("Max Workers", min_value=1, value=st.session_state.max_workers)
st.session_state.max_tokens = st.sidebar.number_input("Max Tokens", min_value=1, value=st.session_state.max_tokens)

if 'step' not in st.session_state:
    st.session_state.step = 1

if st.session_state.step == 1:
    st.header("Step 1: Model Selection")
    model_options = {provider.value: llm_provider_fm[provider].get_supported_models() for provider in LLMProvider}
    
    # Declare a multiselect that shows models across categories, including Ollama models with tags
    all_selected_models = st.session_state.get('selected_models', [])

    # Category selection
    category = st.selectbox("Select Model Category", options=model_options.keys())

    # If 'ollama' is selected, show input for model tag
    if category == "ollama":
        ollama_model_tag = st.text_input("Enter Ollama Model Tag", value="")
        if ollama_model_tag:
            # Add Ollama model with the tag to the available models list
            ollama_model = f"ollama/{ollama_model_tag}"
            all_selected_models.append(ollama_model)
    else:
        # Add models from other categories
        models = st.multiselect(f"Select {category} Models", options=model_options[category])
        all_selected_models.extend([f"{category}/{model}" for model in models])
        all_selected_models = list(set(all_selected_models))

    # Always visible multiselect to see and manage all selected models
    st.session_state.selected_models = st.multiselect(
        "Selected Models", 
        options=all_selected_models,
        default=all_selected_models
    )

    if st.button("Next"):
        if not st.session_state.selected_models:
            st.error("Please select at least one model")
            st.stop()
        
        st.session_state.selected_models = all_selected_models
        st.session_state.step = 2
        st.rerun()


elif st.session_state.step == 2:
    st.header("Step 2: Attack Selection")
    attack_modes = {mode.value: attack_handler_fm[mode].description() for mode in FuzzerAttackMode}
    selected_attacks = st.multiselect("Select Attack Modes", options=attack_modes.keys(), format_func=lambda x: f"{x} - {attack_modes[x]}")
    if st.button("List attack extra"):
        if not selected_attacks:
            st.error("Please select at least one attack mode")
            st.stop()
        
        command = ["python", "run.py", "--list-extra"]
        # Add attack modes
        for attack in selected_attacks:
            command.extend(["-a", attack])
        result = subprocess.run(command, capture_output=True, text=True)
        st.code(result.stderr)
    
    st.session_state.selected_attacks = selected_attacks
    st.session_state.extra_params = st.text_area("Extra Attack Parameters (line-separated key values pairs)", placeholder="KEY1=VALUE1\nKEY2=VALUE2")

    col1, col2 = st.columns([1,1])

    with col1:
        if st.button("Back"):
            st.session_state.step = st.session_state.step - 1
            st.rerun()

    with col2:
        if st.button("Next"):
            if not selected_attacks:
                st.error("Please select at least one attack mode")
                st.stop()
            if st.session_state.extra_params:
                try:
                    for kvp in st.session_state.extra_params.split("\n"):
                        if "=" not in kvp:
                            st.error("Invalid extra parameters format")
                            st.stop()
                        k, v = kvp.split("=")
                except:
                    st.error("Invalid extra parameters format")
                    st.stop()

            st.session_state.step = 3
            st.rerun()

elif st.session_state.step == 3:
    st.header("Step 3: Classifier Selection")
    classifiers = {classifier.value: classifiers_fm[classifier].description() for classifier in Classifier}
    selected_classifiers = st.multiselect("Select Classifiers", options=classifiers.keys(), format_func=lambda x: f"{x} - {classifiers[x]}")

    col1, col2 = st.columns([1,1])

    with col1:
        if st.button("Back"):
            st.session_state.step = st.session_state.step - 1
            st.rerun()

    with col2:
        if st.button("Next"):
            st.session_state.selected_classifiers = selected_classifiers
            st.session_state.step = 4
            st.rerun()

elif st.session_state.step == 4:
    st.header("Step 4: Prompt selection")
    prompt = st.text_area("Enter prompt")

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Back"):
            st.session_state.step = st.session_state.step - 1
            st.rerun()

    with col2:
        if st.button("Next"):
            st.session_state.prompt = prompt
            st.session_state.step = 5
            st.rerun()

elif st.session_state.step == 5:
    st.header("Step 5: Execution")
    command = ["python", "run.py", "-d", st.session_state.db_address, "-w", str(st.session_state.max_workers), "-N", str(st.session_state.max_tokens)]
    if st.session_state.verbose:
        command.append("-v")

    for model in st.session_state.selected_models:
        command.extend(["-m", model])

    for attack in st.session_state.selected_attacks:
        command.extend(["-a", attack])

    for classifier in st.session_state.selected_classifiers:
        command.extend(["-c", classifier])

    ep = {}
    if st.session_state.extra_params:
        for kvp in st.session_state.extra_params.split("\n"):
            k, v = kvp.split("=")
            ep[k] = v

    for k, v in ep.items():
        command.extend(["-e", f"{k}={v}"])

    command.extend(["-t", f"{st.session_state.prompt}"])


    st.code(" ".join(command))
    st.subheader("Edit before executing")
    new_command = st.text_input("command", " ".join(command))
    
    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        if st.button("Back"):
            st.session_state.step = st.session_state.step - 1
            st.rerun()
    with col2:
        run_button = st.button("Run")
    with col3:
        if st.button("Restart"):
            st.session_state.step = 1
            st.rerun()
    
    if run_button:
        env = os.environ.copy()
        env.update(st.session_state.env_vars)
        try:
            idx = new_command.split(" ").index("-t")
            all_args = new_command.split(" ")[:idx+1]
            all_args.append(" ".join(new_command.split(" ")[idx+1:]))
            result = subprocess.run(all_args, capture_output=True, text=True, env=env)
            st.code(result.stdout + result.stderr)
        except Exception as e:
            st.error(f"Error: {str(e)}")