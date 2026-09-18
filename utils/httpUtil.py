import json
import logging
import time
from typing import Any

import requests

__all__ = [
    "HttpRequestError",
    "httpReqWithJsonResp",
]

_logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_WAIT_TIME = 5
_TIMEOUT = 60
_SUCCESS_CODES = {0, "0"}


class HttpRequestError(RuntimeError):
    """Raised when an HTTP JSON API call fails."""


def _serialize_body(j_params: Any) -> str | None:
    if j_params is None:
        return None
    if isinstance(j_params, (bytes, bytearray)):
        return j_params.decode("utf-8")
    if isinstance(j_params, str):
        return j_params
    return json.dumps(j_params)


def _should_retry_status(status_code: int) -> bool:
    return status_code >= 500 or status_code == 429


def _wait_before_retry(attempt: int) -> None:
    if attempt < _MAX_RETRIES - 1:
        time.sleep(_WAIT_TIME)


def _unwrap_api_payload(j_res: Any, method: str, url: str) -> Any:
    """Return business data from either an envelope or a raw JSON payload.

    Envelope responses use {code, message, data|result}: code 0/"0" returns
    data if present, otherwise result. Other codes raise HttpRequestError.
    Responses without a code field are treated as the payload itself.
    """
    if not isinstance(j_res, dict) or "code" not in j_res:
        return j_res

    code = j_res["code"]
    if code not in _SUCCESS_CODES:
        message = j_res.get("message", "")
        msg = (
            f"business error from {method} {url}: "
            f"code={code}, message={message}"
        )
        _logger.error(msg)
        raise HttpRequestError(msg)
    if j_res.get("data") is not None:
        return j_res["data"]
    if "result" in j_res:
        return j_res["result"]
    return j_res.get("data")


def httpReqWithJsonResp(url: str, j_params: Any = None, method: str = "GET") -> Any:
    """Call a JSON API and return its business payload.

    j_params is serialized as a JSON body when provided.

    If the JSON object contains a code field, it is treated as an envelope:
    success (0 or "0") returns data if present, otherwise result; non-zero
    code raises immediately. If code is absent, the decoded JSON is returned
    as-is.

    HTTP 5xx/429 and transport errors are retried up to 3 times. HTTP 4xx and
    invalid JSON raise HttpRequestError immediately without retry.
    """
    headers = {}
    body = _serialize_body(j_params)
    if body is not None:
        headers["Content-Type"] = "application/json"
    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        _logger.info("request %s %s attempt=%s", method, url, attempt)
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                data=body,
                timeout=_TIMEOUT,
            )
            if not response.ok:
                msg = (
                    f"HTTP {response.status_code} from {method} {url}: "
                    f"{response.text[:500]}"
                )
                _logger.error(msg)
                last_error = HttpRequestError(msg)
                if not _should_retry_status(response.status_code):
                    raise last_error
                _wait_before_retry(attempt)
                continue

            try:
                j_res = json.loads(response.content)
            except json.JSONDecodeError as exc:
                msg = f"invalid JSON from {method} {url}: {response.text[:500]}"
                _logger.error(msg)
                raise HttpRequestError(msg) from exc

            return _unwrap_api_payload(j_res, method, url)
        except HttpRequestError:
            raise
        except requests.exceptions.Timeout as exc:
            msg = f"timeout on {method} {url}"
            _logger.error("%s, waiting to retry", msg)
            last_error = HttpRequestError(msg)
            last_error.__cause__ = exc
            _wait_before_retry(attempt)
        except requests.exceptions.RequestException as exc:
            msg = f"request failed for {method} {url}: {exc}"
            _logger.error(msg)
            last_error = HttpRequestError(msg)
            last_error.__cause__ = exc
            _wait_before_retry(attempt)

    detail = f": {last_error}" if last_error else ""
    raise HttpRequestError(
        f"request failed after {_MAX_RETRIES} retries: {method} {url}{detail}"
    ) from last_error
