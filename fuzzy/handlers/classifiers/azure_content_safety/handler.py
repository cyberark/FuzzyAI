# type: ignore

import logging
import os
from enum import Enum
from typing import Any

import aiohttp

from fuzzy.handlers.classifiers.base import BaseClassifier, classifiers_fm
from fuzzy.handlers.classifiers.enums import Classifier

logger = logging.getLogger(__name__)


class AzureContentSafetyModes(str, Enum):
    CONTENT_MODERATION = "content_moderation"  # Scans text for sexual content, violence, hate, and self harm with multi-severity levels
    JAILBREAK_DETECTION = "jailbreak_detection"  # Scans text for Jailbreak attacks are User Prompts designed to provoke the model into exhibiting behaviors it was directed to avoid or to break the rules set in the System Message
    PROMPT_INJECTION = "prompt_injection"  # Scans text for User Prompt injection attacks, where users deliberately exploit system vulnerabilities to elicit unauthorized behavior from the LLM


AZURE_CONTENT_MODERATION_MODE_ACTION_MAP = {
    AzureContentSafetyModes.CONTENT_MODERATION: "analyze",
    AzureContentSafetyModes.JAILBREAK_DETECTION: "detectJailbreak",
    AzureContentSafetyModes.PROMPT_INJECTION: "shieldPrompt",
}

AZURE_CONTENT_MODERATION_MODE_PAYLOAD_KEY_MAP = {
    AzureContentSafetyModes.CONTENT_MODERATION: "text",
    AzureContentSafetyModes.JAILBREAK_DETECTION: "text",
    AzureContentSafetyModes.PROMPT_INJECTION: "userPrompt",
}


@classifiers_fm.flavor(Classifier.AZURE_CONTENT_SAFETY)
class AzureContentSafetyClassifier(BaseClassifier):
    """
    Classifier which uses the Azure Content Safety feature
    """

    def __init__(
        self,
        content_safety_endpoint: str = None,
        content_safety_key: str = None,
        mode: AzureContentSafetyModes = AzureContentSafetyModes.CONTENT_MODERATION,
        **extra: Any,
    ) -> None:
        super().__init__(**extra)

        content_safety_key = content_safety_key or os.environ.get("CONTENT_SAFETY_KEY")
        content_safety_endpoint = content_safety_endpoint or os.environ.get("CONTENT_SAFETY_ENDPOINT")

        if not content_safety_endpoint or not content_safety_key:
            raise ValueError("Content Safety endpoint and key are required")

        self._name = "AZURE_CONTENT_SAFETY_CLASSIFIER"
        self._headers = {"Content-Type": "application/json", "Cache-Control": "no-cache", "Ocp-Apim-Subscription-Key": content_safety_key}
        self._content_safety_endpoint = content_safety_endpoint
        self._mode = mode
        self._async_content_safety_session = aiohttp.ClientSession(headers=self._headers)

    def is_jailbreak(self, value: Any) -> bool:
        result, _ = value
        return result == "Y"
    
    def _get_azure_content_safety_url(self, action: str, version: str = "2024-02-15-preview") -> str:
        return f"{self._content_safety_endpoint}/contentsafety/text:{action}?api-version={version}"

    async def _classify(self, text: str, **extra: Any) -> Any:
        logger.debug(f"Classifier <{self._name}> classifying text: {text}")
        try:
            mode = extra.get("mode", self._mode)
            action = AZURE_CONTENT_MODERATION_MODE_ACTION_MAP.get(mode)
            if not action:
                raise ValueError(f"Invalid mode: {mode}")
            payload_key = AZURE_CONTENT_MODERATION_MODE_PAYLOAD_KEY_MAP.get(mode)
            payload = {payload_key: text}
            if mode == AzureContentSafetyModes.CONTENT_MODERATION:
                payload["outputType"] = "EightSeverityLevels"
            url = self._get_azure_content_safety_url(action=action)

            async with self._async_content_safety_session.post(url, json=payload) as response:
                response_json = await response.json()
                response_detected = None
                
                match mode:
                    case AzureContentSafetyModes.CONTENT_MODERATION:
                        categories = response_json.get("categoriesAnalysis", [])
                        detection_data = [cat for cat in categories if cat.get("severity") > 0]
                        response_detected = len(detection_data) > 0
                    case AzureContentSafetyModes.JAILBREAK_DETECTION:    
                        detection_data = response_json.get("jailbreakAnalysis", {})
                        response_detected = detection_data.get("detected", False)
                    case AzureContentSafetyModes.PROMPT_INJECTION:
                        detection_data = response_json.get("userPromptAnalysis", {})
                        response_detected = detection_data.get("detected", False)
                    case _:
                        raise ValueError(f"Invalid mode: {mode}")

            if response_detected:
                logger.debug(f"Classifier <{self._name}> classified as harmful")
            else:
                logger.debug(f"Classifier <{self._name}> classified as not harmful")

            result = "Y" if response_detected else "N"
            return result, detection_data
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")

    def _sync_classify(self, text: str, **extra: Any) -> Any:
        raise NotImplementedError("Sync classify is not supported for Azure Content Safety Classifier")
