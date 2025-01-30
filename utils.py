import subprocess

def run_ollama_list_command():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Error running 'ollama list': {result.stderr}")
        return
    except FileNotFoundError:
        print("Error: 'ollama' command not found. Please make sure to download ollama from ollama.com")
        return
    except Exception as e:
        print(f"An error occurred while running 'ollama list': {e}")
        return


def run_ollama_pull_command(model: str) -> bool:
    try:
        confirmation = input(f" Model {model} was not found. Do you want to pull the model from ollama? (y/n): ").lower()
    except UnicodeDecodeError:
        # Fall back to a simpler input method if encoding issues occur
        confirmation = input(b" Model {model} was not found. Do you want to pull the model from ollama? (y/n): ".decode('ascii')).lower()

    if confirmation != 'y':
        return False

    try:
        print("Pulling model...please wait")
        # Set encoding explicitly for subprocess
        result = subprocess.run(['ollama', 'pull', model],
                              capture_output=True,
                              text=True,
                              encoding='utf-8')
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"Error running 'ollama pull': {result.stderr}")
            return False
    except FileNotFoundError:
        print("Error: 'ollama' command not found. Please make sure to download ollama from ollama.com")
        return False
    except Exception as e:
        print(f"An error occurred while running 'ollama pull': {e}")
        return False