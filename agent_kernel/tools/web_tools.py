"""WebSearch、WebFetch、URL 安全、抓取缓存和内容后处理。

WebSearch 本身不绑定搜索供应商；``context.web_search_handler`` 接收 query 和域名过滤，
返回结果后由本模块规范化标题、URL、snippet、耗时与查询元数据。缺少 handler 时给出
明确工具错误，而不是偷偷使用其他网络实现。

WebFetch 先验证 HTTP(S) URL、升级普通 HTTP、识别预批准 host，并禁止未经确认的跨域
重定向。默认 urllib 抓取限制响应大小、根据 content-type 解码，HTML 经保守 parser
转为 Markdown-like 文本。随后可交给 apply handler 或次级 ModelProvider，按用户 prompt
提取相关内容。相同 URL 使用短期内存 cache，redirect 结果仍重新走安全检查。

两个工具都声明只读/并发安全，但网络访问仍可能需要 ask；权限建议和实际 callback 在
统一工具管线中处理。提示词常量描述模型何时使用搜索或抓取，不应随实现重写。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
import inspect
import json
import os
import time
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ..messages import AssistantMessage, ToolResultBlock, create_user_message
from ..permissions import PermissionDecision
from .base import Tool, ToolResult, ToolUseContext, ValidationResult


MAX_URL_LENGTH = 2000
MAX_HTTP_CONTENT_LENGTH = 10 * 1024 * 1024
FETCH_TIMEOUT_SECONDS = 60
MAX_REDIRECTS = 10
MAX_MARKDOWN_LENGTH = 100_000
CACHE_TTL_SECONDS = 15 * 60


PREAPPROVED_HOSTS = {
    "platform.claude.com",
    "code.claude.com",
    "modelcontextprotocol.io",
    "github.com/anthropics",
    "agentskills.io",
    "docs.python.org",
    "en.cppreference.com",
    "docs.oracle.com",
    "learn.microsoft.com",
    "developer.mozilla.org",
    "go.dev",
    "pkg.go.dev",
    "www.php.net",
    "docs.swift.org",
    "kotlinlang.org",
    "ruby-doc.org",
    "doc.rust-lang.org",
    "www.typescriptlang.org",
    "react.dev",
    "angular.io",
    "vuejs.org",
    "nextjs.org",
    "expressjs.com",
    "nodejs.org",
    "bun.sh",
    "jquery.com",
    "getbootstrap.com",
    "tailwindcss.com",
    "d3js.org",
    "threejs.org",
    "redux.js.org",
    "webpack.js.org",
    "jestjs.io",
    "reactrouter.com",
    "docs.djangoproject.com",
    "flask.palletsprojects.com",
    "fastapi.tiangolo.com",
    "pandas.pydata.org",
    "numpy.org",
    "www.tensorflow.org",
    "pytorch.org",
    "scikit-learn.org",
    "matplotlib.org",
    "requests.readthedocs.io",
    "jupyter.org",
    "laravel.com",
    "symfony.com",
    "wordpress.org",
    "docs.spring.io",
    "hibernate.org",
    "tomcat.apache.org",
    "gradle.org",
    "maven.apache.org",
    "asp.net",
    "dotnet.microsoft.com",
    "nuget.org",
    "blazor.net",
    "reactnative.dev",
    "docs.flutter.dev",
    "developer.apple.com",
    "developer.android.com",
    "keras.io",
    "spark.apache.org",
    "huggingface.co",
    "www.kaggle.com",
    "www.mongodb.com",
    "redis.io",
    "www.postgresql.org",
    "dev.mysql.com",
    "www.sqlite.org",
    "graphql.org",
    "prisma.io",
    "docs.aws.amazon.com",
    "cloud.google.com",
    "kubernetes.io",
    "www.docker.com",
    "www.terraform.io",
    "www.ansible.com",
    "vercel.com/docs",
    "docs.netlify.com",
    "devcenter.heroku.com",
    "cypress.io",
    "selenium.dev",
    "docs.unity.com",
    "docs.unrealengine.com",
    "git-scm.com",
    "nginx.org",
    "httpd.apache.org",
}


HOSTNAME_ONLY = {host for host in PREAPPROVED_HOSTS if "/" not in host}
PATH_PREFIXES: dict[str, list[str]] = {}
for entry in PREAPPROVED_HOSTS:
    if "/" not in entry:
        continue
    host, path = entry.split("/", 1)
    PATH_PREFIXES.setdefault(host, []).append("/" + path)


@dataclass
class CacheEntry:
    """封装 ``CacheEntry`` 对应的Web 工具状态与行为。"""
    timestamp: float
    data: dict[str, Any]


URL_CACHE: dict[str, CacheEntry] = {}


class NoRedirectHandler(HTTPRedirectHandler):
    """实现 ``NoRedirectHandler`` 所需的协议适配行为。"""
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802
        """禁止 urllib 自动跟随重定向，以便调用方自行验证目标。"""
        return None


class MarkdownishHTMLParser(HTMLParser):
    """无第三方依赖的保守 HTML 文本提取器。"""
    def __init__(self) -> None:
        """初始化实例内部状态和后续处理所需的缓存。"""
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """处理 HTML 开始标签并维护文本结构状态。"""
        if tag in {"script", "style", "noscript"}:
            self.skip_depth += 1
        elif tag in {"p", "div", "section", "article", "header", "footer", "br", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")
        elif tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        """处理 HTML 结束标签并补充分隔符。"""
        if tag in {"script", "style", "noscript"} and self.skip_depth:
            self.skip_depth -= 1
        elif tag in {"p", "div", "section", "article", "header", "footer", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        """收集 HTML 文本节点内容。"""
        if self.skip_depth:
            return
        text = " ".join(data.split())
        if text:
            self.parts.append(text)

    def text(self) -> str:
        """返回清理并合并后的 Markdown-like 文本。"""
        lines = [" ".join(line.split()) for line in "".join(self.parts).splitlines()]
        return "\n".join(line for line in lines if line).strip()


def clear_web_fetch_cache() -> None:
    """清理web fetch 缓存，供Web 工具流程使用。"""
    URL_CACHE.clear()


def get_local_month_year() -> str:
    """获取local month year，供Web 工具流程使用。"""
    override = os.environ.get("CLAUDE_CODE_OVERRIDE_DATE")
    date = datetime.fromisoformat(override) if override else datetime.now()
    return date.strftime("%B %Y")


def web_search_prompt() -> str:
    """完成 ``web_search_prompt`` 对应的Web 工具内部步骤。"""
    current_month_year = get_local_month_year()
    return f"""
- Allows Claude to search the web and use the results to inform responses
- Provides up-to-date information for current events and recent data
- Returns search result information formatted as search result blocks, including links as markdown hyperlinks
- Use this tool for accessing information beyond Claude's knowledge cutoff
- Searches are performed automatically within a single API call

CRITICAL REQUIREMENT - You MUST follow this:
  - After answering the user's question, you MUST include a "Sources:" section at the end of your response
  - In the Sources section, list all relevant URLs from the search results as markdown hyperlinks: [Title](URL)
  - This is MANDATORY - never skip including sources in your response
  - Example format:

    [Your answer here]

    Sources:
    - [Source Title 1](https://example.com/1)
    - [Source Title 2](https://example.com/2)

Usage notes:
  - Domain filtering is supported to include or block specific websites
  - Web search is only available in the US

IMPORTANT - Use the correct year in search queries:
  - The current month is {current_month_year}. You MUST use this year when searching for recent information, documentation, or current events.
  - Example: If the user asks for "latest React docs", search for "React documentation" with the current year, NOT last year
"""


WEB_FETCH_DESCRIPTION = """
- Fetches content from a specified URL and processes it using an AI model
- Takes a URL and a prompt as input
- Fetches the URL content, converts HTML to markdown
- Processes the content with the prompt using a small, fast model
- Returns the model's response about the content
- Use this tool when you need to retrieve and analyze web content

Usage notes:
  - IMPORTANT: If an MCP-provided web fetch tool is available, prefer using that tool instead of this one, as it may have fewer restrictions.
  - The URL must be a fully-formed valid URL
  - HTTP URLs will be automatically upgraded to HTTPS
  - The prompt should describe what information you want to extract from the page
  - This tool is read-only and does not modify any files
  - Results may be summarized if the content is very large
  - Includes a self-cleaning 15-minute cache for faster responses when repeatedly accessing the same URL
  - When a URL redirects to a different host, the tool will inform you and provide the redirect URL in a special format. You should then make a new WebFetch request with the redirect URL to fetch the content.
  - For GitHub URLs, prefer using the gh CLI via Bash instead (e.g., gh pr view, gh issue view, gh api).
"""


def web_fetch_prompt() -> str:
    """完成 ``web_fetch_prompt`` 对应的Web 工具内部步骤。"""
    return f"""IMPORTANT: WebFetch WILL FAIL for authenticated or private URLs. Before using this tool, check if the URL points to an authenticated service (e.g. Google Docs, Confluence, Jira, GitHub). If so, look for a specialized MCP tool that provides authenticated access.
{WEB_FETCH_DESCRIPTION}"""


def make_secondary_model_prompt(markdown_content: str, prompt: str, is_preapproved_domain: bool) -> str:
    """构造secondary model 提示词，供Web 工具流程使用。"""
    guidelines = (
        "Provide a concise response based on the content above. Include relevant details, code examples, and documentation excerpts as needed."
        if is_preapproved_domain
        else """Provide a concise response based only on the content above. In your response:
 - Enforce a strict 125-character maximum for quotes from any source document. Open Source Software is ok as long as we respect the license.
 - Use quotation marks for exact language from articles; any language outside of the quotation should never be word-for-word the same.
 - You are not a lawyer and never comment on the legality of your own prompts and responses.
 - Never produce or reproduce exact song lyrics."""
    )
    return f"""
Web page content:
---
{markdown_content}
---

{prompt}

{guidelines}
"""


def is_preapproved_host(hostname: str, pathname: str) -> bool:
    """判断preapproved 主机，供Web 工具流程使用。"""
    if hostname in HOSTNAME_ONLY:
        return True
    for prefix in PATH_PREFIXES.get(hostname, []):
        if pathname == prefix or pathname.startswith(prefix + "/"):
            return True
    return False


def is_preapproved_url(url: str) -> bool:
    """判断preapproved URL，供Web 工具流程使用。"""
    try:
        parsed = urlparse(url)
        return is_preapproved_host(parsed.hostname or "", parsed.path or "/")
    except ValueError:
        return False


def validate_url(url: str) -> bool:
    """只允许带有效 hostname 的 HTTP(S) URL。"""
    if len(url) > MAX_URL_LENGTH:
        return False
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.username or parsed.password:
        return False
    hostname = parsed.hostname or ""
    if len(hostname.split(".")) < 2:
        return False
    return True


def upgrade_http_to_https(url: str) -> str:
    """升级HTTP to https，供Web 工具流程使用。"""
    parsed = urlparse(url)
    if parsed.scheme != "http":
        return url
    return urlunparse(("https", parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))


def is_permitted_redirect(original_url: str, redirect_url: str) -> bool:
    """判断permitted 重定向，供Web 工具流程使用。"""
    try:
        original = urlparse(original_url)
        redirect = urlparse(redirect_url)
    except ValueError:
        return False
    if redirect.scheme != original.scheme or redirect.port != original.port:
        return False
    if redirect.username or redirect.password:
        return False
    strip_www = lambda hostname: hostname.removeprefix("www.") if hostname else ""
    return strip_www(original.hostname) == strip_www(redirect.hostname)


def status_text(status_code: int) -> str:
    """完成 ``status_text`` 对应的Web 工具内部步骤。"""
    if status_code == 301:
        return "Moved Permanently"
    if status_code == 308:
        return "Permanent Redirect"
    if status_code == 307:
        return "Temporary Redirect"
    if status_code == 302:
        return "Found"
    return "OK" if 200 <= status_code < 300 else "HTTP Error"


def _html_to_text(content: str) -> str:
    """完成 ``_html_to_text`` 对应的Web 工具内部步骤。"""
    parser = MarkdownishHTMLParser()
    parser.feed(content)
    return parser.text()


def _decode_response(raw: bytes, content_type: str) -> str:
    """完成 ``_decode_response`` 对应的Web 工具内部步骤。"""
    text = raw.decode("utf-8", errors="replace")
    if "text/html" in content_type:
        return _html_to_text(text)
    return text


def get_web_fetch_user_agent() -> str:
    """获取web fetch 用户 agent，供Web 工具流程使用。"""
    return "Claude-User (claude-code-python-port; +https://support.anthropic.com/)"


def get_url_markdown_content(url: str) -> dict[str, Any]:
    """执行抓取、重定向验证、解码和 HTML 转文本，并写入内存 cache。"""
    if not validate_url(url):
        raise ValueError("Invalid URL")

    cached = URL_CACHE.get(url)
    if cached and (time.monotonic() - cached.timestamp) < CACHE_TTL_SECONDS:
        return dict(cached.data)

    upgraded_url = upgrade_http_to_https(url)
    current_url = upgraded_url
    opener = build_opener(NoRedirectHandler)
    for _ in range(MAX_REDIRECTS + 1):
        request = Request(
            current_url,
            headers={
                "Accept": "text/markdown, text/html, */*",
                "User-Agent": get_web_fetch_user_agent(),
            },
            method="GET",
        )
        try:
            with opener.open(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
                raw = response.read(MAX_HTTP_CONTENT_LENGTH + 1)
                if len(raw) > MAX_HTTP_CONTENT_LENGTH:
                    raise ValueError("Fetched content exceeds maximum size")
                content_type = response.headers.get("content-type", "")
                data = {
                    "bytes": len(raw),
                    "code": response.status,
                    "codeText": getattr(response, "reason", "") or status_text(response.status),
                    "content": _decode_response(raw, content_type),
                    "contentType": content_type,
                    "url": url,
                }
                URL_CACHE[url] = CacheEntry(time.monotonic(), dict(data))
                return data
        except HTTPError as exc:
            if exc.code not in {301, 302, 307, 308}:
                raise
            location = exc.headers.get("location")
            if not location:
                raise ValueError("Redirect missing Location header") from exc
            redirect_url = urljoin(current_url, location)
            if not is_permitted_redirect(current_url, redirect_url):
                return {
                    "type": "redirect",
                    "originalUrl": current_url,
                    "redirectUrl": redirect_url,
                    "statusCode": exc.code,
                }
            current_url = redirect_url
    raise ValueError(f"Too many redirects (exceeded {MAX_REDIRECTS})")


def _normalise_search_results(result: Any, query: str, duration_seconds: float) -> dict[str, Any]:
    """完成 ``_normalise_search_results`` 对应的Web 工具内部步骤。"""
    if isinstance(result, dict) and "results" in result:
        output = dict(result)
        output.setdefault("query", query)
        output.setdefault("durationSeconds", duration_seconds)
        return output
    if isinstance(result, str):
        return {"query": query, "results": [result], "durationSeconds": duration_seconds}
    if isinstance(result, list):
        hits = []
        for item in result:
            if isinstance(item, dict):
                hits.append({"title": str(item.get("title", "")), "url": str(item.get("url", ""))})
        return {
            "query": query,
            "results": [{"tool_use_id": "web_search", "content": hits}] if hits else [],
            "durationSeconds": duration_seconds,
        }
    return {"query": query, "results": [], "durationSeconds": duration_seconds}


async def _maybe_call(handler: Callable[..., Any], *args: Any) -> Any:
    """完成 ``_maybe_call`` 对应的Web 工具内部步骤。"""
    result = handler(*args)
    if inspect.isawaitable(result):
        return await result
    return result


async def _apply_prompt_to_markdown(
    context: ToolUseContext,
    prompt: str,
    markdown_content: str,
    is_preapproved_domain: bool,
) -> str:
    """应用提示词 to markdown，供Web 工具流程使用。"""
    model_prompt = make_secondary_model_prompt(markdown_content, prompt, is_preapproved_domain)
    provider = getattr(context, "model_provider", None)
    if provider is None:
        return model_prompt
    model = getattr(context, "web_fetch_model", None) or "fake-model"
    async for assistant_message in provider.stream(
        messages=[create_user_message(model_prompt)],
        system_prompt=[],
        tools=[],
        options={
            "model": model,
            "querySource": "web_fetch_apply",
        },
    ):
        for block in assistant_message["message"]["content"]:
            if block["type"] == "text":
                return block["text"]
    return "No response from model"


class WebSearchTool(Tool):
    """带域名 allow/block filter 的搜索工具。"""
    name = "WebSearch"
    search_hint = "search the web for current information"
    max_result_size_chars = 100_000
    input_schema = {"query": str, "allowed_domains": list, "blocked_domains": list}
    required_fields = ("query",)

    async def description(self, input: dict | None = None) -> str:
        """返回提供给模型和调用方展示的工具简短说明。"""
        if input and isinstance(input.get("query"), str):
            return f"Claude wants to search the web for: {input['query']}"
        return "Claude wants to search the web"

    async def prompt(self) -> str:
        """返回该工具按源码保留的模型使用提示词。"""
        return web_search_prompt()

    def user_facing_name(self, input: dict | None = None) -> str:
        """根据当前输入返回适合界面展示的工具名称。"""
        return "Web Search"

    def is_concurrency_safe(self, input: dict) -> bool:
        """判断当前输入是否可以与相邻安全工具并发执行。"""
        return True

    def is_read_only(self, input: dict) -> bool:
        """判断当前输入是否只读取外部状态而不产生修改。"""
        return True

    async def validate_input(self, input: dict, context: ToolUseContext) -> ValidationResult:
        """执行依赖上下文的业务输入校验。"""
        query = input.get("query")
        allowed_domains = input.get("allowed_domains")
        blocked_domains = input.get("blocked_domains")
        if not isinstance(query, str) or len(query) < 2:
            return ValidationResult(False, "Error: Missing query", 1)
        if allowed_domains and blocked_domains:
            return ValidationResult(
                False,
                "Error: Cannot specify both allowed_domains and blocked_domains in the same request",
                2,
            )
        for field_name in ("allowed_domains", "blocked_domains"):
            value = input.get(field_name)
            if value is not None and not all(isinstance(item, str) for item in value):
                return ValidationResult(False, f"Field {field_name} has invalid type.", 3)
        return ValidationResult(True)

    async def check_permissions(self, input: dict, context: ToolUseContext) -> PermissionDecision:
        """返回该工具针对当前输入的初步权限建议。"""
        return PermissionDecision.ask("WebSearchTool requires permission.")

    async def call(
        self,
        args: dict,
        context: ToolUseContext,
        can_use_tool,
        parent_message: AssistantMessage,
        on_progress=None,
    ) -> ToolResult:
        """执行工具核心逻辑，并返回标准 ToolResult。"""
        start = time.perf_counter()
        query = args["query"]
        if on_progress:
            on_progress({"type": "query_update", "query": query})
        handler = getattr(context, "web_search_handler", None)
        if handler is None:
            raise RuntimeError("WebSearch is not configured. Provide a web_search_handler or configure the local runner search provider.")
        raw_result = await _maybe_call(handler, args)
        duration_seconds = time.perf_counter() - start
        output = _normalise_search_results(raw_result, query, duration_seconds)
        if on_progress:
            result_count = 0
            for item in output.get("results") or []:
                if isinstance(item, dict):
                    result_count += len(item.get("content") or [])
            on_progress({"type": "search_results_received", "resultCount": result_count, "query": query})
        return ToolResult(output)

    def map_tool_result_to_tool_result_block_param(self, output: dict, tool_use_id: str) -> ToolResultBlock:
        """把工具内部结果映射为 Anthropic tool_result block。"""
        query = output.get("query", "")
        results = output.get("results") or []
        formatted_output = f'Web search results for query: "{query}"\n\n'
        for result in results:
            if result is None:
                continue
            if isinstance(result, str):
                formatted_output += result + "\n\n"
            elif isinstance(result, dict):
                content = result.get("content") or []
                if content:
                    formatted_output += f"Links: {json.dumps(content, ensure_ascii=False, separators=(',', ':'))}\n\n"
                else:
                    formatted_output += "No links found.\n\n"
        formatted_output += "\nREMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks."
        return {"tool_use_id": tool_use_id, "type": "tool_result", "content": formatted_output.strip()}


class WebFetchTool(Tool):
    """抓取单个 URL，并根据 prompt 返回相关内容。"""
    name = "WebFetch"
    search_hint = "fetch and extract content from a URL"
    max_result_size_chars = 100_000
    input_schema = {"url": str, "prompt": str}
    required_fields = ("url", "prompt")

    async def description(self, input: dict | None = None) -> str:
        """返回提供给模型和调用方展示的工具简短说明。"""
        if input and isinstance(input.get("url"), str):
            try:
                hostname = urlparse(input["url"]).hostname
                if hostname:
                    return f"Claude wants to fetch content from {hostname}"
            except ValueError:
                pass
        return "Claude wants to fetch content from this URL"

    async def prompt(self) -> str:
        """返回该工具按源码保留的模型使用提示词。"""
        return web_fetch_prompt()

    def user_facing_name(self, input: dict | None = None) -> str:
        """根据当前输入返回适合界面展示的工具名称。"""
        return "Fetch"

    def is_concurrency_safe(self, input: dict) -> bool:
        """判断当前输入是否可以与相邻安全工具并发执行。"""
        return True

    def is_read_only(self, input: dict) -> bool:
        """判断当前输入是否只读取外部状态而不产生修改。"""
        return True

    async def validate_input(self, input: dict, context: ToolUseContext) -> ValidationResult:
        """执行依赖上下文的业务输入校验。"""
        url = input.get("url")
        if not isinstance(url, str) or not validate_url(url):
            return ValidationResult(
                False,
                f'Error: Invalid URL "{url}". The URL provided could not be parsed.',
                1,
                {"reason": "invalid_url"},
            )
        prompt = input.get("prompt")
        if not isinstance(prompt, str) or prompt == "":
            return ValidationResult(False, "prompt is required.", 2)
        return ValidationResult(True)

    async def check_permissions(self, input: dict, context: ToolUseContext) -> PermissionDecision:
        """返回该工具针对当前输入的初步权限建议。"""
        url = input.get("url", "")
        if isinstance(url, str):
            parsed = urlparse(url)
            if is_preapproved_host(parsed.hostname or "", parsed.path or "/"):
                return PermissionDecision.allow()
        return PermissionDecision.ask("Claude requested permissions to use WebFetch, but you haven't granted it yet.")

    async def call(
        self,
        args: dict,
        context: ToolUseContext,
        can_use_tool,
        parent_message: AssistantMessage,
        on_progress=None,
    ) -> ToolResult:
        """执行工具核心逻辑，并返回标准 ToolResult。"""
        start = time.perf_counter()
        url = args["url"]
        prompt = args["prompt"]
        if on_progress:
            on_progress({"message": f"Fetching {url}"})
        fetch_handler = getattr(context, "web_fetch_handler", None)
        response = await _maybe_call(fetch_handler, url) if fetch_handler is not None else get_url_markdown_content(url)

        if isinstance(response, dict) and response.get("type") == "redirect":
            status_code = int(response.get("statusCode") or 302)
            code_text = status_text(status_code)
            message = f'''REDIRECT DETECTED: The URL redirects to a different host.

Original URL: {response.get("originalUrl")}
Redirect URL: {response.get("redirectUrl")}
Status: {status_code} {code_text}

To complete your request, I need to fetch content from the redirected URL. Please use WebFetch again with these parameters:
- url: "{response.get("redirectUrl")}"
- prompt: "{prompt}"'''
            return ToolResult(
                {
                    "bytes": len(message.encode("utf-8")),
                    "code": status_code,
                    "codeText": code_text,
                    "result": message,
                    "durationMs": int((time.perf_counter() - start) * 1000),
                    "url": url,
                }
            )

        content = str(response.get("content", ""))
        content_type = str(response.get("contentType") or response.get("content_type") or "")
        bytes_count = int(response.get("bytes") or len(content.encode("utf-8")))
        code = int(response.get("code") or 200)
        code_text = str(response.get("codeText") or status_text(code))
        is_preapproved = is_preapproved_url(url)

        if is_preapproved and "text/markdown" in content_type and len(content) < MAX_MARKDOWN_LENGTH:
            result = content
        else:
            truncated = content if len(content) <= MAX_MARKDOWN_LENGTH else content[:MAX_MARKDOWN_LENGTH] + "\n\n[Content truncated due to length...]"
            apply_handler = getattr(context, "web_fetch_apply_handler", None)
            if apply_handler is not None:
                result = await _maybe_call(apply_handler, prompt, truncated, is_preapproved)
            else:
                result = await _apply_prompt_to_markdown(context, prompt, truncated, is_preapproved)

        persisted_path = response.get("persistedPath") or response.get("persisted_path")
        if persisted_path:
            persisted_size = int(response.get("persistedSize") or response.get("persisted_size") or bytes_count)
            result += f"\n\n[Binary content ({content_type}, {persisted_size} bytes) also saved to {persisted_path}]"

        return ToolResult(
            {
                "bytes": bytes_count,
                "code": code,
                "codeText": code_text,
                "result": result,
                "durationMs": int((time.perf_counter() - start) * 1000),
                "url": url,
            }
        )

    def map_tool_result_to_tool_result_block_param(self, content: dict, tool_use_id: str) -> ToolResultBlock:
        """把工具内部结果映射为 Anthropic tool_result block。"""
        return {"tool_use_id": tool_use_id, "type": "tool_result", "content": content["result"]}
