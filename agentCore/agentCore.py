import os, sys
import logging
from openai import OpenAI

__all__ = ["callLLM"]

def callLLM() -> str:
    return "hello world"