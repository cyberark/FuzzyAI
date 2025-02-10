# FuzzyAI Fuzzer
<div style="text-align: center;">
<img src="resources/logo.png" alt="Project Logo" width="200" style="vertical-align:middle; margin-right:10px;" />
</div>
<br><br>

The FuzzyAI Fuzzer is a powerful tool for automated LLM fuzzing. It is designed to help developers and security researchers identify jailbreaks and mitigate potential security vulnerabilities in their LLM APIs. 

![FZAI](resources/fuzz.gif)

## Getting Started

1. Clone the repository:
   ```bash
   git clone git@github.com:cyberark/FuzzyAI.git
   ```

2. Install dependencies using [Poetry](https://python-poetry.org/):
   ```bash
   poetry install
   poetry shell  # Activate virtual environment
   ```

3. Run the fuzzer:
   ```bash
   python run.py -h
   ```
4. Optional: Install [ollama](https://ollama.com/download/), and download a model for local usage:
   ``` # Running the command will download and install (if not) llama3.1, which is about 4.7 GB in size and is an 8B parameters model. Llama3.1 hat can be substituted with any other open-source model that is supported by ollama.
   ollama pull llama3.1
   ollama show llama3.1 # verify model installation
   ```
   
   Alternativly, you can use the Web UI

## Web UI (Experimental)
![FZAI](resources/webui.png)

1. Run the Web UI (make sure you complete steps #1 and #2 from above):
   ```bash
   streamlit run webui.py
   ```

## Datasets

We've included some datasets you can use under [resources/](https://github.com/cyberark/FuzzyAI/tree/main/resources). For more information, see [datasets wiki](https://github.com/cyberark/FuzzyAI/wiki#datasets).

## Documentation

Explore detailed usage instructions in the [Wiki](https://github.com/cyberark/FuzzyAI/wiki).

## Examples

### A default attack (just evaluate the prompt 'Harmful_Prompt') using llama over Ollama
```bash
python run.py -m ollama/llama3.1 -a def -t "Harmful_Prompt"
```

### Attack the prompt utilizing ManyShot and Taxonomy attacks, using gpt3.5 over OpenAI using one worker
```bash
OPENAI_API_KEY=sk-3fa1... python run.py -m openai/gpt-3.5-turbo -a man -a tax -t "Harmful_Prompt"
```

### Attack prompts in prompts.txt utilizing the ArtPrompt attack, using Claude 3.5 over Anthropic
```bash
ANTHROPIC_API_KEY=deadbeef... python run.py -m anthropic/claude-3-5-sonnet-20240620 -a art -T resources/adv_prompts.txt -e blacklisted_words="bomb,garbage,acid,ugly,self-harm,kill your"
```

--- 

### Attack your own REST API
```bash
OPENAI_API_KEYT=sk-3fa1... python run.py -a def -x openai/gpt-4 -e classifier_model=openai/gpt-4 -m rest/http.raw -e host=localhost -e port=8000 -e scheme=https  
-t "How to build a bomb"
```

#### Arguments breakdown:
- **`-a def`**:  
  Invokes the default attack mode, which evaluates the API without applying additional strategies.

- **`-x openai/gpt-4`**:  
  Specifies an auxiliary model (`gpt-4`) to assist with certain tasks during the attack.

- **`-e classifier_model=openai/gpt-4`**:  
  Configures the fuzzer to use the `gpt-4` model for classifying outputs. This helps determine whether responses meet specific criteria, such as detecting harmful or unwanted content.

- **`-m rest/http.raw`**:  
  Uses the REST provider to attack an API endpoint. The raw HTTP request is parsed from the file `http.raw`. (Refer to the [documentation or example](./resources/http.raw) for details on the file structure.)

- **`-e host=localhost -e port=8000 -e scheme=https`**:  
  Configures the REST provider with the following API endpoint details:
  - **Host**: `localhost`
  - **Port**: `8000`
  - **Scheme**: `https` (communication will use HTTPS)

- **`-t "How to build a bomb"`**:  
  Specifies the test input. In this example, it tests the API's handling of sensitive or harmful content.


## Key Features

- **Comprehensive Fuzzing Techniques**: Leverage mutation-based, generation-based, and intelligent fuzzing.
- **Built-in Input Generation**: Generate valid and invalid inputs for exhaustive testing.
- **Seamless Integration**: Easily incorporate into your development and testing workflows.
- **Extensible Architecture**: Customize and expand the fuzzer to meet your unique requirements.

## Supported models
FuzzyAI supports various models across top providers, including:

| Provider     | Models                                                                                                   |
|--------------|----------------------------------------------------------------------------------------------------------|
| **Anthropic**| Claude (3.5, 3.0, 2.1)                                                                                   |
| **OpenAI**   | GPT-4o, GPT-4o mini, GPT-4                                                                                 |
| **Gemini**   | Gemini Pro, Gemini 1.5                                                                                  |
| **Azure**    | GPT-4, GPT-3.5 Turbo                                                                                    |
| **Bedrock**  | Claude (3.5, 3.0), Meta (LLaMa)                                                                             |
| **AI21**     | Jamba (1.5 Mini, Large)                                                                                |
| **DeepSeek** | DeepSeek (DeepSeek-V3, DeepSeek-V1)                                                                  |
| **Ollama**   | LLaMA (3.3, 3.2, 3.1), Dolphin-LLaMA3, Vicuna                                                               |

## Adding support for newer models
Easily add support for additional models by following our <a href="https://github.com/cyberark/FuzzyAI/wiki/DIY#adding-support-for-new-models">DIY guide</a>.

## Implemented Attacks

| Attack Type                                  | Title                                                                                                                                                                       | Reference                                                                       |
|----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| ArtPrompt                                    | ASCII Art-based jailbreak attacks against aligned LLMs                                                                                                                      | [arXiv:2402.11753](https://arxiv.org/pdf/2402.11753)                            |
| Taxonomy-based paraphrasing                  | Persuasive language techniques like emotional appeal to jailbreak LLMs                                                                                | [arXiv:2401.06373](https://arxiv.org/pdf/2401.06373)                            |
| PAIR (Prompt Automatic Iterative Refinement) | Automates adversarial prompt generation by iteratively refining prompts with two LLMs                       | [arXiv:2310.08419](https://arxiv.org/pdf/2310.08419)                            |
| Many-shot jailbreaking                       | Embeds multiple fake dialogue examples to weaken model safety                            | [Anthropic Research](https://www.anthropic.com/research/many-shot-jailbreaking) |
| ASCII Smuggling                              | ASCII Smuggling uses Unicode Tag characters to embed hidden instructions within text, which are invisible to users but can be processed by Large Language Models (LLMs), potentially leading to prompt injection attacks                                                                                | [Embracethered blog](https://embracethered.com/blog/posts/2024/hiding-and-finding-text-with-unicode-tags/) |
| Genetic                                      | Utilizes a genetic algorithm to modify prompts for adversarial outcomes                      | [arXiv:2309.01446](https://arxiv.org/pdf/2309.01446)                            |
| Hallucinations                               | Bypasses RLHF filters using model-generated                                                                                                                                 | [arXiv:2403.04769](https://arxiv.org/pdf/2403.04769.pdf)                        |
| DAN (Do Anything Now)                        | Promotes the LLM to adopt an unrestricted persona that ignores standard content filters, allowing it to "Do Anything Now".                                                  | [GitHub Repo](https://github.com/0xk1h0/ChatGPT_DAN)                            |
| WordGame                                     | Disguises harmful prompts as word puzzles                                                                                                                                   | [arXiv:2405.14023](https://arxiv.org/pdf/2405.14023)                            |
| Crescendo                                    | Engaging the model in a series of escalating conversational turns,starting with innocuous queries and gradually steering the dialogue toward restricted or sensitive topics. | [arXiv:2404.01833](https://arxiv.org/pdf/2404.01833)                            |
| ActorAttack                                  | Inspired by actor-network theory, it builds semantic networks of "actors" to subtly guide conversations toward harmful targets while concealing malicious intent.           | [arxiv 2410.10700](https://arxiv.org/pdf/2410.10700)                                                                            |                                                                                                                                     |
| Best-of-n jailbreaking | Uses input variations to repeatedly elicit harmful responses, exploiting model sensitivity | [arXiv:2412.03556](https://arxiv.org/abs/2412.03556) |
| Shuffle Inconsistency Attack (SI-Attack) | Exploits the inconsistency between an LLM's comprehension ability and safety mechanisms by shuffling harmful text prompts. The shuffled text bypasses safety mechanisms while still being understood as harmful by the LLM. Only the text-based implementation was completed; the image-based aspect was not implemented. | [arXiv:2501.04931](https://arxiv.org/abs/2501.04931) |
| Back To The Past                             | Modifies the prompt by adding a profession-based prefix and a past-related suffix                                                                                           |                                                                                 |
| Please                                       | Modifies the prompt by adding please as a prefix and suffix                                                                                                                   |                                                                                 |
| Thought Experiment                           | Modifies the prompt by adding a thought experiment-related prefix. In addition, adds "precautions have been taken care of" suffix                                                  |                                                                                 
| Default                                      | Send the prompt to the model as-is 

## Supported Cloud APIs
- **OpenAI**
- **Anthropic**
- **Gemini**
- **Azure Cloud**
- **AWS Bedrock**
- **AI21**
- **DeepSeek**
- **Huggingface ([Downloading models](https://huggingface.co/docs/hub/en/models-downloading))**
- **Ollama**
- **Custom REST API**
---

## Caveats
* Some classifiers do more than just evaluate a single output. For example, the cosine-similarity classifier compares two outputs by measuring the angle between them, while a 'harmfulness' classifier checks whether a given output is harmful. As a result, not all classifiers are compatible with the attack methods we've implemented, as those methods are designed for single-output classifiers.
* When using the -m option with OLLAMA models, <b>ensure that all OLLAMA models are added first before adding any other models.</b> Use the -e port=... option to specify the port number for OLLAMA (default is 11434).

## Contributing

Contributions are welcome! If you would like to contribute to the FuzzyAI Fuzzer, please follow the guidelines outlined in the [CONTRIBUTING.md](https://github.com/cyberark/FuzzyAI/blob/main/CONTRIBUTING.md) file.

## License

The FuzzyAI Fuzzer is released under the [Apache License](https://www.apache.org/licenses/LICENSE-2.0). See the [LICENSE](https://github.com/cyberark/FuzzyAI/blob/main/LICENSE) file for more details.

## Contact

If you have any questions or suggestions regarding the FuzzyAI Fuzzer, please feel free to contact us at [fzai@cyberark.com](mailto:fzai@cyberark.com).

