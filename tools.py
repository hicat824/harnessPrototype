import json
import logging
import os

tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "查询指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "需要查询天气的城市，例如深圳、北京、上海",
                }
            },
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "list_local_files",
        "description": "列举本地文件列表",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            },
            "required": ["path"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "read_meta_info",
        "description": "读取文件的metaInfo",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            },
            "required": ["path"],
            "additionalProperties": False
        },
        "strict": True
    }
]

def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        return _getWeather(**arguments)
    elif name == "list_local_files":
        return _listLocalFiles(**arguments)
    elif name == "read_meta_info":
        return _readFileMetaInfo(**arguments)

    raise ValueError(f"Unknown tool: {name}")

def _getWeather(city: str) -> str:
    print(f"[tool] get_weather(city={city})")

    weather_data = {
        "北京": "晴，28°C",
        "上海": "多云，30°C",
        "深圳": "阵雨，32°C",
    }

    return weather_data.get(city, "暂无天气数据")


def _listLocalFiles(path: str) -> str:
    res = []
    for root, dirs, files in os.walk(path):
        for file in files:
           res.append(os.path.join(root, file))

    print(f'在 {path} 下扫描发现 {len(res)} 个文件')
    return json.dumps(res, ensure_ascii=False)

def _readFileMetaInfo(path: str) -> str:
    if not os.path.exists(path):
        return None
    with open(path, "r") as fp:
        content = json.load(fp)
        if "metaInfo" not in content:
            return None

    return json.dumps(content["metaInfo"], ensure_ascii=False)