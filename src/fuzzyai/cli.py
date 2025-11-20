import argparse
import asyncio
import json
import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import aiofiles
import aiofiles.os
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from fuzzyai.consts import DEFAULT_SYSTEM_PROMPT, PARAMETER_MAX_TOKENS, WIKI_LINK
from fuzzyai.fuzzer import Fuzzer
from fuzzyai.handlers.attacks.base import attack_handler_fm
from fuzzyai.handlers.attacks.enums import FuzzerAttackMode
from fuzzyai.handlers.classifiers.base import classifiers_fm
from fuzzyai.handlers.classifiers.enums import Classifier
from fuzzyai.llm.providers.base import llm_provider_fm
from fuzzyai.llm.providers.enums import LLMProvider
from fuzzyai.utils.custom_logging_formatter import CustomFormatter
from fuzzyai.utils.utils import CURRENT_TIMESTAMP, generate_report, print_report, run_ollama_list_command

logging.basicConfig(level=logging.INFO)

load_dotenv()

banner = """\033[31m
   ______  ____________  ___ ___   ____
  / __/ / / /_  /_  /\\ \\/ (_) _ | /  _/
 / _// /_/ / / /_/ /_ \\  / / __ |_/ /  
/_/  \\____/ /___/___/ /_(_)_/ |_/___/  
                                       
\033[0m"""

root_logger = logging.getLogger()
root_logger.handlers.clear()
console_handler = logging.StreamHandler()
console_handler.setFormatter(CustomFormatter())
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

class LoadFromFile (argparse.Action):
    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace, 
                 values: Any, option_string: Optional[str] = None) -> None:
        cli_args: str = str()

        with values as f:
            # parse arguments in the file and store them in the target namespace
            json_data = json.loads(f.read())
            for k,v in json_data.items():
                if isinstance(v, list):
                    for item in v:
                        cli_args += f"--{k} {item} "
                elif isinstance(v, bool):
                    if v:
                        cli_args += f"--{k} "
                else:
                    cli_args += f"--{k} {v} "

            parser.parse_args(shlex.split(cli_args), namespace)

async def run_fuzzer(args: argparse.Namespace) -> None:
    """
    Run the fuzzer
    """
    if args.verbose:
        logger.info('Verbose logging ON')
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger().propagate = True

    if args.ollama_list:
        run_ollama_list_command()
        return

    for req_arg in ['attack_modes']:
        if not getattr(args, req_arg):
            logger.error(f"Required argument: --{req_arg}")
            return

    for attack_mode in args.attack_modes:
            if attack_mode not in set(FuzzerAttackMode):
                logger.error(f"Attack method '{attack_mode}' not found")
                return
            
    if hasattr(args, 'list_extra') and args.list_extra:
        for attack_mode in args.attack_modes:
            attack_handler_cls = attack_handler_fm[FuzzerAttackMode(attack_mode)]
            logger.info("Extra arguments can be defined by using -e key=value, i.e -e max_tokens=200")
            logger.info(f"List of extra arguments for {attack_mode}:\n{json.dumps(attack_handler_cls.extra_args(), indent=2)}")

        return
    
    for req_arg in ['model']:
        if not getattr(args, req_arg):
            logger.error(f"Required argument: --{req_arg}")
            return
        
    if not args.target_prompt and not args.target_prompts_file:
        logger.error('Please provide a target prompt (-t) or a file with target prompts (-T)')
        return
    
    try:
        if hasattr(args, 'extra') and args.extra:
            extra = {k:v for k,v in (x.split('=') for x in args.extra)}
            del args.extra
            extra.update(**vars(args))
        else:
            extra = vars(args)
    except Exception:
        raise ValueError(f"Error adding extra argument, please make sure you use the correct format, i.e -e key=value. For further help, please check the wiki: {WIKI_LINK}")

    if hasattr(args, 'target_prompts_file') and args.target_prompts_file:
        with open(args.target_prompts_file, 'r') as f:
            prompts = f.readlines()
        prompts = [prompt.strip() for prompt in prompts if prompt.strip()]
    else:
        prompts = args.target_prompt

    if hasattr(args, 'benign_prompts') and args.benign_prompts:
        logger.info(f"Adding {args.benign_prompts} benign prompts to the attack")
        with open('resources/benign_prompts.txt', 'r') as f:
            benign_prompts = f.readlines()[:args.benign_prompts]

        prompts += [prompt.strip() for prompt in benign_prompts if prompt.strip()]

    fuzzer = Fuzzer(db_address=args.db_address)
    
    if hasattr(args, 'classifier') and args.classifier:
        for classifier in args.classifier:
            if classifier not in set(Classifier):
                logger.error(f"Classifier {classifier} not found")
                return
            
            logger.debug(f"Adding classifier {classifier}")
            fuzzer.add_classifier(classifiers_fm[classifier](**extra))
    else:
        logger.info('No classifier defined - using default HAR classifier ("Is the following content harmful?")')
        fuzzer.add_classifier(classifiers_fm[Classifier.HARMFUL_LLM](**extra))
    
    extra_copy = extra.copy()
    extra_copy.pop("model")

    args.model = [x.lower() for x in args.model]
    args.auxiliary_model = [x.lower() for x in args.auxiliary_model]

    if args.max_workers < len(args.model):
        logger.warning("Number of workers is less than number of models. Is this intentional? This may cause slower performance.")

    for seed, qualified_model_name in enumerate(args.model + args.auxiliary_model):
        provider_seed: int = seed if args.max_workers > 1 else 0
        fuzzer.add_llm(qualified_model_name, seed=provider_seed, **extra_copy)
    
    if hasattr(args, 'classifier_model') and args.classifier_model:
        if args.classifier_model not in args.model + args.auxiliary_model:
            logger.info('Adding auxiliary model for classifier: ' + args.classifier_model)
            fuzzer.add_llm(args.classifier_model, **extra_copy)

    extra['model'] = set(extra['model']) # Remove duplicates

    try:
        report, raw_results = await fuzzer.fuzz(prompts=prompts, **extra)
    except Exception as e:
        logger.error(f"Error during attack: {str(e)}.\nFor further help, please check the wiki: {WIKI_LINK}")
        await fuzzer.cleanup()
        return

    if isinstance(args.model, (list, set)) and args.model:
        model_str = next(iter(args.model))  
    else:
        model_str = str(args.model)

    model_name = model_str.split("/")[-1]
    
    folder = f'results/{model_name}/{attack_mode+'__'+CURRENT_TIMESTAMP}'

    await aiofiles.os.makedirs(folder, exist_ok=True)

    if report.attacking_techniques and any(atp.total_prompts_count > 0 for atp in report.attacking_techniques):
        if raw_results:
            logger.info(f"Dumping raw results to {folder}/raw.jsonl")
            async with aiofiles.open(f"{folder}/raw.jsonl", 'w', encoding="utf-8") as f:
                for raw_result in raw_results:    
                    await f.write(raw_result.model_dump_json())

        if report:
            logger.info(f"Dumping results to {folder}/report.json")
            async with aiofiles.open(f"{folder}/report.json", 'w', encoding="utf-8") as f:
                await f.write(report.model_dump_json())
            
            generate_report(report)
            print_report(report)

async def run_webui(args: argparse.Namespace) -> None:
    """
    Run the web UI
    """
    port = 8080
    
    process = subprocess.Popen(
        ["streamlit", "run", "src/fuzzyai/webui.py", "--server.port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    await asyncio.sleep(2)
    print("Web UI is running at http://localhost:8080, Use Ctrl+C to exit")
    process.wait()

async def run_cli() -> None:
    print(banner)

    attack_methods_help: str = str()
    models_help: str = str()
    classifiers_help: str = str()

    parser = argparse.ArgumentParser(prog="fuzzyai", formatter_class=argparse.RawTextHelpFormatter, description="fuzzyai - Automatic LLM Fuzzer")
    subparsers = parser.add_subparsers(dest='command', required=True)

    webui_fuzzer = subparsers.add_parser('webui', help='Run the web UI')
    webui_fuzzer.set_defaults(func=run_webui)

    fuzz_parser = subparsers.add_parser('fuzz', help='Run the fuzzer')
    fuzz_parser.set_defaults(func=run_fuzzer)

    fuzz_parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    fuzz_parser.add_argument('-d', '--db_address', help='MongoDB address (default: 127.0.0.1)', type=str, default="127.0.0.1")
    fuzz_parser.add_argument('-w', '--max_workers', help='Max workers (default: 1)', type=int, default=1)
    fuzz_parser.add_argument('-i', '--attack_id', help='Load previous attack id', type=str, default=None)
    fuzz_parser.add_argument('-C', '--configuration_file', help='Load fuzzer arguments from JSON configuration file', type=open, action=LoadFromFile)

    # create models help string
    models: dict[LLMProvider, list[str]] = {}
    for provider in LLMProvider:
        supported_models = llm_provider_fm[provider].get_supported_models()
        if isinstance(supported_models, str):
            models.setdefault(provider, []).append(supported_models)
            continue
        for model in llm_provider_fm[provider].get_supported_models():
            models.setdefault(provider, []).append(model)
    
    for provider_name, model_name in models.items():
        for model in model_name:
            models_help += f"{provider_name.value}/{model}\n"
        
        models_help += "\n"

    fuzz_parser.add_argument('-m', '--model', help=f'Model(s) to attack, any of:\n\n{models_help}', 
                        action="append", type=str, default=[])
    # Add each attack method to the attack
    for attack_method in FuzzerAttackMode:
        attack_methods_help += f"{attack_method.value}\t{attack_handler_fm[attack_method].description().strip()}\n"

    fuzz_parser.add_argument('-a', '--attack_modes', help=f'Add attack mode any of:\n\n{attack_methods_help}', action="append", 
                        type=str, default=[])

    # Create classifiers help
    for classifier in Classifier:
        classifiers_help += f"{classifier.value}\t{classifiers_fm[classifier].description().strip()}\n"
    
    fuzz_parser.add_argument('-c', '--classifier', help=f'Add a classifier (default: har), any of:\n\n{classifiers_help}', action="append", type=str, default=[])
    fuzz_parser.add_argument('-cm', '--classifier_model', help=f'Defines which model to use for classification (default: use the attacked model)', type=str, default=None)
    fuzz_parser.add_argument('-tc', '--truncate-cot', help='Remove CoT (Chain Of Thought) when classifying results (default: true)', action='store_true', default=True)
    fuzz_parser.add_argument('-N', f'--{PARAMETER_MAX_TOKENS}', help='Max tokens to generate when generating LLM response (default: 100)', 
                        type=int, default=100)

    fuzz_parser.add_argument('-b', f'--benign_prompts', help='Adds n benign prompts to the attack (default: 0)', 
                        type=int, default=0)
    
    group = fuzz_parser.add_mutually_exclusive_group()
    group.add_argument('-t', '--target-prompt', help='Prompt to attack (One or more)', action="append", type=str, default=[])
    group.add_argument('-T', '--target-prompts-file', help='Prompts to attack (from file, line separated)', type=str, default=None)

    fuzz_parser.add_argument('-s', '--system-prompt', help=f'System prompt to use (default: {DEFAULT_SYSTEM_PROMPT}', type=str, default=DEFAULT_SYSTEM_PROMPT)
    fuzz_parser.add_argument('-e', '--extra', help='Extra parameters (for providers/attack handlers) in form of key=value', action="append", 
                        type=str, default=[])
    fuzz_parser.add_argument('-E', '--list-extra', help='List extra arguments for selected attack method(s)', action='store_true', default=False)
    fuzz_parser.add_argument('-x', '--auxiliary-model', help=f'Add auxiliary models which can be used in the attack',
                        action="append", type=str, default=[])
    fuzz_parser.add_argument('-I', '--improve-attempts', help='Attempts to refine the LLM response up to n times following a successful jailbreak. Default: 0 (no refinement attempts)', type=int, default=0)
    fuzz_parser.add_argument('-ol', '--ollama-list', action='store_true',  # Modified to use store_true
                        help='Shows all the ollama models that are installed on the station')
    
    args = parser.parse_args()
    if asyncio.iscoroutinefunction(args.func):
        await args.func(args)
    else:
        args.func(args)

def main() -> None:
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt as e:
        logger.info('Exiting...')
        exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=False)
        exit(1)

if __name__ == "__main__":
    main()