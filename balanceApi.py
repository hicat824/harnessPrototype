import os
import requests
from openai import OpenAI

_DEEPSEEK_GET_BALANCE_URL = "https://api.deepseek.com/user/balance"

def getUsrBalance():
    """
    Get the user's balance from the DeepSeek API.
    """
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set.")

    payload={}
    headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {api_key}'
    }

    response = requests.get(_DEEPSEEK_GET_BALANCE_URL, headers=headers, data=payload)
    if response.status_code != 200:
        raise Exception(f"Failed to get balance: {response.status_code} - {response.text}")

    j_balance = response.json()
    try:
        balance_infos = j_balance.get("balance_infos")
        balance = balance_infos[0].get("total_balance")
    except Exception as e:
        raise Exception(f"Error parsing balance from response: {e}")

    return f'当前账户余额为 {balance} 元'
