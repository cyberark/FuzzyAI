# type: ignore
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from fuzzyai.enums import EnvironmentVariables
from fuzzyai.handlers.attacks.base import attack_handler_fm
from fuzzyai.handlers.attacks.enums import FuzzerAttackMode
from fuzzyai.handlers.classifiers.base import classifiers_fm
from fuzzyai.handlers.classifiers.enums import Classifier
from fuzzyai.llm.providers.base import llm_provider_fm
from fuzzyai.llm.providers.enums import LLMProvider
from fuzzyai.utils.utils import get_ollama_models

load_dotenv()

st.set_page_config(
    page_title="FuzzyAI Web UI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.image("resources/logo.png", width=175)

defaults = {
    "env_vars": {},
    "verbose": False,
    "db_address": "127.0.0.1",
    "max_workers": 1,
    "max_tokens": 1000,
    "truncate_cot": True,
    "extra_params": {},
    "selected_models": [],
    "selected_models_aux": [],
    "selected_attacks": [],
    "selected_classifiers": [],
    "classifier_model": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value
    
st.sidebar.header("Environment Settings")
api_keys = [x.value for x in EnvironmentVariables]
new_env_key = st.sidebar.selectbox("Name", options=api_keys)
new_env_value = st.sidebar.text_input("Value")
if st.sidebar.button("Add Variable"):
    if new_env_key and new_env_value:
        st.session_state.env_vars[new_env_key] = new_env_value

for x in EnvironmentVariables:
    if x.value in os.environ:
        st.session_state.env_vars[x.value] = os.environ[x.value]
        
# Create a container for the table
with st.sidebar.container():
    if st.session_state.env_vars:
        # Create three columns for key, value, and delete button
        cols = st.columns([2, 2, 1])
        
        # Headers
        cols[0].markdown("**Key**")
        cols[1].markdown("**Value**")
        cols[2].markdown("**Action**")
        
        # Display each variable in a row
        for key, value in dict(st.session_state.env_vars).items():
            col1, col2, col3 = st.columns([2, 2, 1])
            col1.text(key)
            #masked_value = '*' * len(value) if 'key' in key.lower() or 'token' in key.lower() else value
            masked_value = value[:8] + "..."
            col2.text(masked_value)
            if col3.button("❌", key=f"delete_{key}"):
                del st.session_state.env_vars[key]
                st.rerun()

st.sidebar.header("Classifier Model")
if st.session_state.selected_models_aux:
    classifier_model = st.sidebar.selectbox(
        "Select Classifier Model (optional)",
        options=st.session_state.selected_models_aux,
        index=None if st.session_state.classifier_model is None 
        else st.session_state.selected_models_aux.index(st.session_state.classifier_model)
    )
    st.session_state.classifier_model = classifier_model
else:
    st.sidebar.selectbox(
        "Select Classifier Model (optional)",
        options=["No aux models available"],
        disabled=True
    )
    st.session_state.classifier_model = None

st.sidebar.header("Fuzzy settings")
st.session_state.verbose = st.sidebar.checkbox("Verbose Logging", value=st.session_state.verbose)
st.session_state.db_address = st.sidebar.text_input("MongoDB Address", value=st.session_state.db_address)
st.session_state.max_workers = st.sidebar.number_input("Max Workers", min_value=1, value=st.session_state.max_workers)
st.session_state.max_tokens = st.sidebar.number_input("Max Tokens", min_value=1, value=st.session_state.max_tokens)


if 'step' not in st.session_state:
    st.session_state.step = 1

if st.session_state.step == 1:
    ollama_models: list[str] = []

    def on_model_select(category, select_key, models: str):
        def on_change():
            st.session_state[models].append(f"{category}/{st.session_state[select_key]}")
        return on_change
    
    st.header("Step 1: Model Selection")
    st.subheader("Select target models for the attack")
    model_options = {provider.value: llm_provider_fm[provider].get_supported_models() for provider in LLMProvider}
    
    # Category selection
    category = st.selectbox("Select Model Category", options=model_options.keys(), index=None)

    # If 'ollama' is selected, show input for model tag
    if category == "ollama":
        ollama_models = get_ollama_models()
        model_options[category] = ollama_models

    if category:
        st.selectbox(f"Select {category} Models", options=model_options[category], index=None, 
                        key='model', on_change=on_model_select(category, 'model', 'selected_models'))

    # Always visible multiselect to see and manage all selected models
    st.session_state.selected_models = st.multiselect(
        "Selected Models", 
        options=st.session_state.selected_models,
        default=st.session_state.selected_models
    )

    st.subheader("Select auxiliary models")
    st.markdown("Auxiliary models are optional and can be used for additional tasks such as classification or other purposes. If you don't need any auxiliary models, you can skip this selection.")
    # Category selection
    category_aux = st.selectbox("Select Model Category", options=model_options.keys(), key="cat_aux", index=None)

    if category_aux == "ollama" and not ollama_models:
        model_options[category] = get_ollama_models()

    if category_aux:
        st.selectbox(f"Select {category_aux} Models", options=model_options[category_aux], 
                        index=None, key='model_aux', on_change=on_model_select(category_aux, 'model_aux', 'selected_models_aux'))

    # Always visible multiselect to see and manage all selected models
    st.session_state.selected_models_aux = st.multiselect(
        "Selected Auxiliary Models", 
        options=st.session_state.selected_models_aux,
        default=st.session_state.selected_models_aux
    )

    if st.button("Next"):
        if not st.session_state.selected_models:
            st.error("Please select at least one model")
            st.stop()
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
        
        command = ["fuzzyai", "fuzz", "--list-extra"]
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
    command = ["fuzzyai", "fuzz"]
    
    if st.session_state.db_address != defaults["db_address"]:
        command.extend([
            "-d", st.session_state.db_address
        ])
    if st.session_state.max_workers != defaults["max_workers"]:
        command.extend([
            "-w", str(st.session_state.max_workers)
        ])
    if st.session_state.max_tokens != defaults["max_tokens"]:
        command.extend([
            "-N", str(st.session_state.max_tokens)
        ])
        
    if st.session_state.verbose:
        command.append("-v")

    for model in list(set(st.session_state.selected_models)):
        command.extend(["-m", model])

    for model in list(set(st.session_state.selected_models_aux)):
        command.extend(["-x", model])

    for attack in st.session_state.selected_attacks:
        command.extend(["-a", attack])

    for classifier in st.session_state.selected_classifiers:
        command.extend(["-c", classifier])

    if st.session_state.classifier_model:
        command.extend(["--classifier-model", st.session_state.classifier_model])

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
