"""Untrusted remote media fetching with SSRF and resource-exhaustion guards.

DNS is resolved once, every answer must be globally routable, and the HTTP/TLS
connection is pinned to one of those validated addresses.  Pinning matters: a
validate-then-open flow lets an attacker return a public address for validation
and a loopback/private address for the HTTP library's second DNS lookup.
"""
from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
import time
import urllib.parse

_REDIRECTS = {301, 302, 303, 307, 308}
_MAX_REDIRECTS = 5
_MAX_URL_CHARS = 8192


class RemoteFetchError(RuntimeError):
    """A remote URL was unsafe, invalid, too large, or could not be fetched."""


def _parsed_url(url: str) -> urllib.parse.SplitResult:
    if len(url) > _MAX_URL_CHARS:
        raise RemoteFetchError("remote media URL is too long")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise RemoteFetchError("remote media URL must use http:// or https://")
    if parsed.username is not None or parsed.password is not None:
        raise RemoteFetchError("remote media URL must not contain credentials")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise RemoteFetchError("remote media URL has an invalid port") from exc
    return parsed


def resolve_public_addresses(parsed: urllib.parse.SplitResult) -> list[str]:
    """Resolve a URL host once and reject the complete answer set if any IP is private."""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise RemoteFetchError(f"remote media host could not be resolved: {exc}") from exc
    addresses: list[str] = []
    for info in infos:
        raw = str(info[4][0]).split("%", 1)[0]
        try:
            address = ipaddress.ip_address(raw)
        except ValueError as exc:
            raise RemoteFetchError("remote media resolved to an invalid address") from exc
        if not address.is_global:
            raise RemoteFetchError(
                f"remote media must resolve only to public addresses ({address} is blocked)"
            )
        if raw not in addresses:
            addresses.append(raw)
    if not addresses:
        raise RemoteFetchError("remote media host did not resolve to an address")
    return addresses


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
        super().__init__(host, port, timeout=timeout)
        self._address = address

    def connect(self) -> None:
        self.sock = socket.create_connection((self._address, self.port), self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
        self._tls_context = ssl.create_default_context()
        super().__init__(host, port, timeout=timeout, context=self._tls_context)
        self._address = address

    def connect(self) -> None:
        raw = socket.create_connection((self._address, self.port), self.timeout)
        try:
            # Certificate verification and SNI use the original hostname, while the TCP
            # destination remains the address validated above.
            self.sock = self._tls_context.wrap_socket(raw, server_hostname=self.host)
        except BaseException:
            raw.close()
            raise


def _connection(
    parsed: urllib.parse.SplitResult, address: str, timeout: float,
) -> http.client.HTTPConnection:
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    cls = _PinnedHTTPSConnection if parsed.scheme == "https" else _PinnedHTTPConnection
    return cls(parsed.hostname or "", port, address, timeout)


def _host_header(parsed: urllib.parse.SplitResult) -> str:
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    default_port = 443 if parsed.scheme == "https" else 80
    return f"{host}:{parsed.port}" if parsed.port and parsed.port != default_port else host


def _open_response(parsed, addresses: list[str], deadline: float):
    """Try validated IPv4/IPv6 answers without ever asking DNS a second time."""
    path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    last_error: BaseException | None = None
    for address in addresses:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        conn = _connection(parsed, address, remaining)
        try:
            conn.request(
                "GET", path,
                headers={
                    "Host": _host_header(parsed),
                    "User-Agent": "local-llm-server",
                    "Accept-Encoding": "identity",
                    "Connection": "close",
                },
            )
            return conn, conn.getresponse()
        except (OSError, ValueError, http.client.HTTPException, ssl.SSLError) as exc:
            last_error = exc
            conn.close()
    if last_error is not None:
        raise RemoteFetchError(f"remote media could not be downloaded: {last_error}") from last_error
    raise RemoteFetchError("remote media download timed out")


def fetch_remote(url: str, max_bytes: int, *, timeout: float = 120.0) -> bytes:
    """Fetch an HTTP(S) resource once, bounded by bytes, time, redirects, and public IPs."""
    if max_bytes < 1 or timeout <= 0:
        raise ValueError("max_bytes and timeout must be positive")
    deadline = time.monotonic() + timeout
    current = url
    for redirect_count in range(_MAX_REDIRECTS + 1):
        parsed = _parsed_url(current)
        addresses = resolve_public_addresses(parsed)
        conn, response = _open_response(parsed, addresses, deadline)
        try:
            if response.status in _REDIRECTS:
                location = response.getheader("Location")
                if not location:
                    raise RemoteFetchError("remote media redirect has no Location header")
                if redirect_count >= _MAX_REDIRECTS:
                    raise RemoteFetchError("remote media has too many redirects")
                current = urllib.parse.urljoin(current, location)
                continue
            if not (200 <= response.status < 300):
                raise RemoteFetchError(f"remote media server returned HTTP {response.status}")
            encoding = (response.getheader("Content-Encoding") or "identity").lower()
            if encoding not in ("", "identity"):
                raise RemoteFetchError("compressed remote media responses are not accepted")
            raw_length = response.getheader("Content-Length")
            if raw_length:
                try:
                    declared = int(raw_length)
                except ValueError as exc:
                    raise RemoteFetchError("remote media has invalid Content-Length") from exc
                if declared < 0 or declared > max_bytes:
                    raise RemoteFetchError(f"remote media is too large (> {max_bytes} bytes)")
            chunks: list[bytes] = []
            copied = 0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RemoteFetchError("remote media download timed out")
                if conn.sock is not None:
                    conn.sock.settimeout(remaining)
                chunk = response.read(min(1024 * 1024, max_bytes + 1 - copied))
                if not chunk:
                    return b"".join(chunks)
                copied += len(chunk)
                if copied > max_bytes:
                    raise RemoteFetchError(f"remote media is too large (> {max_bytes} bytes)")
                chunks.append(chunk)
        except RemoteFetchError:
            raise
        except (OSError, ValueError, http.client.HTTPException, ssl.SSLError) as exc:
            raise RemoteFetchError(f"remote media could not be downloaded: {exc}") from exc
        finally:
            conn.close()
    raise RemoteFetchError("remote media has too many redirects")
