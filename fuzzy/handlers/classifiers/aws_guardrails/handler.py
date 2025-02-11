import logging
import os
from typing import Any, Optional

import boto3

from fuzzy.handlers.classifiers.base import BaseClassifier, classifiers_fm
from fuzzy.handlers.classifiers.enums import Classifier

logger = logging.getLogger(__name__)


@classifiers_fm.flavor(Classifier.AWS_GUARDRAILS)
class AWSGuardrailsClassifier(BaseClassifier):
    """
    Classifier which uses the AWS Bedrock Guardrails feature
    """

    def __init__(self, guardrail_id: Optional[str] = None, guardrail_version: Optional[str] = None, 
                 region: Optional[str] = None, **extra: Any) -> None:
        super().__init__(**extra)

        aws_session_requirements = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"]
        # verify that all required AWS session variables are in os.environ
        if not all(var in os.environ for var in aws_session_requirements):
            raise ValueError("AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and AWS_SESSION_TOKEN are required in env")

        if any(x is None for x in [guardrail_id, guardrail_version, region]):
            raise ValueError("guardrial_id, guardrial_version or region are missing, please use -e to define them")
        
        boto3.setup_default_session(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            aws_session_token=os.environ["AWS_SESSION_TOKEN"],
            region_name=region,
        )

        self._name = "AWS_GUARDRAILS_CLASSIFIER"
        self._guardrail_id = guardrail_id
        self._guardrail_version = guardrail_version
        self._bedrock_runtime = boto3.client("bedrock-runtime")

    def is_jailbreak(self, value: Any) -> bool:
        result, _, _ = value
        return bool(result == "GUARDRAIL_INTERVENED")
    
    def _format_text(self, text: str) -> dict[str, dict[str, str]]:
        return {"text": {"text": text}}

    def _sync_classify(self, text: str, **extra: Any) -> Any:
        source: str = extra.get("source", "INPUT")
        logger.debug(f"Classifier <{self._name}> classifying text: {text}")
        formatted_text = self._format_text(text)
        try:
            response = self._bedrock_runtime.apply_guardrail(
                guardrailIdentifier=self._guardrail_id,
                guardrailVersion=self._guardrail_version,
                source=source,
                content=[formatted_text],
            )
            action, outputs, assessments = response["action"], response["outputs"], response["assessments"]
            if len(outputs) > 1:
                raise Exception(f"Guardrail intervened. Output: {outputs[0]['text']}")
            outputs = len(outputs) > 0 and outputs[0]["text"] or None

            if action == "GUARDRAIL_INTERVENED":
                logger.debug(f"Classifier <{self._name}> classified as harmful")
            else:
                logger.debug(f"Classifier <{self._name}> classified as not harmful")

            return action, outputs, assessments
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")

    async def _classify(self, *args: Any, **extra: Any) -> Any:
        raise NotImplementedError
