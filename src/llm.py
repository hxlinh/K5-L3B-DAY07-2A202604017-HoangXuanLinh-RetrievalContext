"""Client tối giản cho OpenAI Chat Completions (urllib, không cần SDK). Lỗi được raise, không fallback."""
import json
import os
import time
import urllib.error
import urllib.request

URL = 'https://api.openai.com/v1/chat/completions'
RETRY_STATUS = {429, 500, 502, 503, 504}      # quá tải / lỗi tạm thời -> thử lại


class OpenAIChat:
    def __init__(self, model=None, api_key=None, timeout=90, retries=3):
        self.model = model or os.environ['MODEL']
        self.key = api_key or os.environ['OPENAI_API_KEY']
        self.timeout, self.retries = timeout, retries

    def complete(self, messages, tools=None, tool_choice='auto', **extra):
        """extra: tham số API bổ sung, ví dụ response_format={'type': 'json_object'}."""
        payload = {'model': self.model, 'messages': messages, **extra}
        if tools:
            payload.update(tools=tools, tool_choice=tool_choice, parallel_tool_calls=False)
        body = json.dumps(payload).encode('utf-8')
        for attempt in range(1, self.retries + 2):
            req = urllib.request.Request(URL, data=body, headers={
                'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'})
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.load(resp)
                break
            except urllib.error.HTTPError as e:
                if e.code in RETRY_STATUS and attempt <= self.retries:
                    time.sleep(2 ** attempt)
                    continue
                try:
                    detail = json.load(e).get('error', {}).get('message', '')[:300]
                except Exception:
                    detail = ''
                raise RuntimeError(f'OpenAI HTTP {e.code}: {detail}') from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt <= self.retries:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f'Không kết nối được OpenAI: {e}') from e
        return {'message': data['choices'][0]['message'], 'usage': data.get('usage', {}),
                'model': data.get('model'), 'latency_ms': round((time.perf_counter() - t0) * 1000),
                'attempts': attempt}
