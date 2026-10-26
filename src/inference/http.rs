//! Blocking HTTP/1.1 adapter for **local, trusted debugging only**.
//!
//! Only plain HTTP on loopback is supported; there is no TLS or authentication.
//! Do not expose this server through a proxy or port forward to untrusted users.
//! Connections are closed after one request. Bodies and headers are each limited
//! to 4 MiB, and the diagnostic server caps generation at 32,768 tokens.

use super::{ChatMessage, Generation, GenerationConfig, InferenceBackend, validate_messages};
use serde::Deserialize;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::time::Duration;

const MAX_BYTES: usize = 4 * 1024 * 1024;
const MAX_SERVER_TOKENS: usize = 32_768;
const SERVER_TIMEOUT: Duration = Duration::from_secs(30);

/// An OpenAI-compatible client restricted to unauthenticated loopback HTTP.
///
/// `base_url` may end in `/v1` or be a server root (including a path prefix).
/// Discovery always checks `/v1/models`: an explicit ID must be advertised;
/// otherwise the first advertised model is selected.
pub struct HttpBackend {
    endpoint: Endpoint,
    model: String,
    timeout: Duration,
}

impl HttpBackend {
    pub fn connect(
        base_url: &str,
        model: Option<String>,
        timeout: Duration,
    ) -> Result<Self, String> {
        if timeout.is_zero() {
            return Err("HTTP timeout must be nonzero".into());
        }
        let endpoint = Endpoint::parse(base_url)?;
        let body = endpoint.request("GET", "models", None, timeout)?;
        #[derive(Deserialize)]
        struct Model {
            id: String,
        }
        #[derive(Deserialize)]
        struct Models {
            data: Vec<Model>,
        }
        let models: Models = serde_json::from_slice(&body)
            .map_err(|e| format!("invalid /v1/models response: {e}"))?;
        if models.data.is_empty() || models.data.iter().any(|m| m.id.trim().is_empty()) {
            return Err("/v1/models returned no usable model IDs".into());
        }
        let model = match model {
            Some(id) if models.data.iter().any(|m| m.id == id) => id,
            Some(id) => {
                return Err(format!(
                    "unknown model ID {id:?}; not advertised by /v1/models"
                ));
            }
            None => models.data[0].id.clone(),
        };
        Ok(Self {
            endpoint,
            model,
            timeout,
        })
    }
}

impl InferenceBackend for HttpBackend {
    fn model_id(&self) -> &str {
        &self.model
    }

    fn generate(
        &mut self,
        messages: &[ChatMessage],
        options: &GenerationConfig,
    ) -> Result<Generation, String> {
        validate_messages(messages)?;
        options.validate()?;
        let mut request = json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": options.max_tokens,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "seed": options.seed,
            "stop": options.stop,
            "stream": false,
        });
        // Qwen servers that support this extension can avoid spending the token
        // budget on thinking. Never substitute reasoning_content for the answer.
        if self.model.to_ascii_lowercase().contains("qwen") {
            request["chat_template_kwargs"] = json!({"enable_thinking": false});
        }
        let body = serde_json::to_vec(&request).map_err(|e| e.to_string())?;
        let body = self
            .endpoint
            .request("POST", "chat/completions", Some(&body), self.timeout)?;
        #[derive(Deserialize)]
        struct Message {
            content: Option<String>,
        }
        #[derive(Deserialize)]
        struct Choice {
            message: Message,
            finish_reason: String,
        }
        #[derive(Default, Deserialize)]
        struct Usage {
            #[serde(default)]
            prompt_tokens: usize,
            #[serde(default)]
            completion_tokens: usize,
        }
        #[derive(Deserialize)]
        struct Completion {
            model: Option<String>,
            choices: Vec<Choice>,
            usage: Option<Usage>,
        }
        let response: Completion = serde_json::from_slice(&body)
            .map_err(|e| format!("invalid chat completion response: {e}"))?;
        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or("chat completion response has no choices")?;
        let text = choice
            .message
            .content
            .filter(|text| !text.trim().is_empty())
            .ok_or("chat completion response has null or empty answer content")?;
        if choice.finish_reason.trim().is_empty() {
            return Err("chat completion response has an empty finish_reason".into());
        }
        let usage = response.usage.unwrap_or_default();
        Ok(Generation {
            text,
            model: response.model.unwrap_or_else(|| self.model.clone()),
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            // In particular, retain "length" so the collector can filter it.
            finish_reason: choice.finish_reason,
        })
    }
}

struct Endpoint {
    authority: String,
    addresses: Vec<SocketAddr>,
    prefix: String,
}

impl Endpoint {
    fn parse(url: &str) -> Result<Self, String> {
        if url.starts_with("https://") {
            return Err("TLS/HTTPS is unsupported; use plain HTTP on loopback only".into());
        }
        let rest = url
            .strip_prefix("http://")
            .ok_or("base URL must use http:// (loopback only; no TLS)")?;
        if rest
            .bytes()
            .any(|b| b.is_ascii_control() || b.is_ascii_whitespace())
            || rest.contains(['?', '#', '@', '\\'])
        {
            return Err(
                "invalid HTTP base URL (credentials, query and fragment are unsupported)".into(),
            );
        }
        let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
        let addresses = loopback_addresses(authority, Some(80))?;
        let path = path.trim_end_matches('/');
        let prefix = if path.is_empty() {
            "/v1".into()
        } else if path == "v1" || path.ends_with("/v1") {
            format!("/{path}")
        } else {
            format!("/{path}/v1")
        };
        Ok(Self {
            authority: authority.into(),
            addresses,
            prefix,
        })
    }

    fn request(
        &self,
        method: &str,
        route: &str,
        body: Option<&[u8]>,
        timeout: Duration,
    ) -> Result<Vec<u8>, String> {
        let body = body.unwrap_or_default();
        if body.len() > MAX_BYTES {
            return Err("HTTP request body exceeds 4 MiB".into());
        }
        let mut last_error = String::new();
        let mut connected = None;
        for address in &self.addresses {
            match TcpStream::connect_timeout(address, timeout) {
                Ok(stream) => {
                    connected = Some(stream);
                    break;
                }
                Err(error) => last_error = error.to_string(),
            }
        }
        let mut stream = connected.ok_or_else(|| format!("HTTP connect failed: {last_error}"))?;
        set_timeouts(&stream, timeout)?;
        let head = format!(
            "{method} {}/{route} HTTP/1.1\r\nHost: {}\r\nAccept: application/json\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.prefix,
            self.authority,
            body.len()
        );
        if head.len() > MAX_BYTES {
            return Err("HTTP request headers exceed 4 MiB".into());
        }
        stream.write_all(head.as_bytes()).map_err(io_error)?;
        stream.write_all(body).map_err(io_error)?;
        let mut reader = BufReader::new(stream);
        // Permit a small, bounded number of informational responses.
        for _ in 0..8 {
            let head = read_head(&mut reader)?;
            let mut words = head.start.split_whitespace();
            let version = words.next().unwrap_or_default();
            let code = words.next().unwrap_or_default();
            if !matches!(version, "HTTP/1.1" | "HTTP/1.0")
                || code.len() != 3
                || !code.bytes().all(|b| b.is_ascii_digit())
            {
                return Err("invalid HTTP response status line".into());
            }
            let status: u16 = code.parse().map_err(|_| "invalid HTTP status code")?;
            if !(100..600).contains(&status) || status == 101 {
                return Err(format!("unsupported HTTP status {status}"));
            }
            if status < 200 {
                continue;
            }
            let body = if status == 204 || status == 304 {
                Vec::new()
            } else {
                read_body(&mut reader, &head, true)?
            };
            if !(200..300).contains(&status) {
                let detail = String::from_utf8_lossy(&body);
                let detail: String = detail.chars().take(2048).collect();
                return Err(format!(
                    "HTTP {status} from {method} {}/{route}: {detail}",
                    self.prefix
                ));
            }
            return Ok(body);
        }
        Err("too many informational HTTP responses".into())
    }
}

fn io_error(error: std::io::Error) -> String {
    format!("HTTP I/O error: {error}")
}

fn set_timeouts(stream: &TcpStream, timeout: Duration) -> Result<(), String> {
    stream.set_read_timeout(Some(timeout)).map_err(io_error)?;
    stream.set_write_timeout(Some(timeout)).map_err(io_error)
}

/// Resolve only an already-validated loopback host. No arbitrary DNS lookup is
/// allowed; even localhost resolution is checked before any connection or bind.
fn loopback_addresses(
    authority: &str,
    default_port: Option<u16>,
) -> Result<Vec<SocketAddr>, String> {
    let (host, port) = if let Some(rest) = authority.strip_prefix('[') {
        let (host, suffix) = rest
            .split_once(']')
            .ok_or("invalid bracketed IPv6 address")?;
        let port = if suffix.is_empty() {
            default_port.ok_or("a listening port is required")?
        } else {
            suffix
                .strip_prefix(':')
                .ok_or("invalid IPv6 authority")?
                .parse::<u16>()
                .map_err(|_| "invalid HTTP port")?
        };
        (host, port)
    } else if let Some((host, port)) = authority.split_once(':') {
        (
            host,
            port.parse::<u16>()
                .map_err(|_| "invalid HTTP port (IPv6 must be bracketed)")?,
        )
    } else {
        (
            authority,
            default_port.ok_or("a listening port is required")?,
        )
    };
    if host.eq_ignore_ascii_case("localhost") {
        let addresses: Vec<_> = ("localhost", port)
            .to_socket_addrs()
            .map_err(io_error)?
            .collect();
        if addresses.is_empty() || addresses.iter().any(|address| !address.ip().is_loopback()) {
            return Err("localhost must resolve exclusively to loopback addresses".into());
        }
        return Ok(addresses);
    }
    let ip: IpAddr = host
        .parse()
        .map_err(|_| "only loopback IPs or localhost are allowed")?;
    if !ip.is_loopback() {
        return Err("only loopback addresses are allowed (no remote HTTP endpoints)".into());
    }
    Ok(vec![SocketAddr::new(ip, port)])
}

struct Head {
    start: String,
    fields: Vec<(String, String)>,
}

// read_until alone can allocate arbitrarily before seeing a newline. Bound the
// allocation as well as the eventual parsed header/body length.
fn read_line(reader: &mut impl BufRead, budget: &mut usize) -> Result<Vec<u8>, String> {
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf().map_err(io_error)?;
        if available.is_empty() {
            return Err("unexpected EOF in HTTP headers or chunk framing".into());
        }
        let length = available
            .iter()
            .position(|&b| b == b'\n')
            .map_or(available.len(), |i| i + 1);
        if length > *budget {
            return Err("HTTP headers or chunk framing exceed 4 MiB".into());
        }
        line.extend_from_slice(&available[..length]);
        reader.consume(length);
        *budget -= length;
        if line.last() == Some(&b'\n') {
            if !line.ends_with(b"\r\n") {
                return Err("HTTP lines must end in CRLF".into());
            }
            line.truncate(line.len() - 2);
            return Ok(line);
        }
    }
}

fn header_field(line: &[u8]) -> Result<(String, String), String> {
    let line = std::str::from_utf8(line).map_err(|_| "invalid HTTP header encoding")?;
    let (name, value) = line.split_once(':').ok_or("invalid HTTP header")?;
    if name.is_empty()
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
        || value.bytes().any(|b| (b < 32 && b != b'\t') || b == 127)
    {
        return Err("invalid HTTP header name or value".into());
    }
    Ok((name.to_ascii_lowercase(), value.trim().into()))
}

fn read_head(reader: &mut impl BufRead) -> Result<Head, String> {
    let mut budget = MAX_BYTES;
    let start = String::from_utf8(read_line(reader, &mut budget)?)
        .map_err(|_| "invalid HTTP start line encoding")?;
    let mut fields = Vec::new();
    loop {
        let line = read_line(reader, &mut budget)?;
        if line.is_empty() {
            return Ok(Head { start, fields });
        }
        fields.push(header_field(&line)?);
    }
}

fn read_body(
    reader: &mut impl BufRead,
    head: &Head,
    eof_delimited: bool,
) -> Result<Vec<u8>, String> {
    let mut content_length = None;
    let mut chunked = false;
    for (name, value) in &head.fields {
        match name.as_str() {
            "content-length" => {
                if content_length.is_some()
                    || value.is_empty()
                    || !value.bytes().all(|b| b.is_ascii_digit())
                {
                    return Err("invalid or duplicate Content-Length".into());
                }
                let length = value
                    .parse::<usize>()
                    .map_err(|_| "invalid Content-Length")?;
                if length > MAX_BYTES {
                    return Err("HTTP body exceeds 4 MiB".into());
                }
                content_length = Some(length);
            }
            "transfer-encoding" => {
                if chunked || !value.eq_ignore_ascii_case("chunked") {
                    return Err(
                        "unsupported or duplicate Transfer-Encoding (only chunked is supported)"
                            .into(),
                    );
                }
                chunked = true;
            }
            _ => {}
        }
    }
    if chunked && content_length.is_some() {
        return Err("ambiguous HTTP framing: both Transfer-Encoding and Content-Length".into());
    }
    if chunked {
        let mut body = Vec::new();
        let mut framing_budget = MAX_BYTES;
        loop {
            let line = read_line(reader, &mut framing_budget)?;
            let size = line.split(|&b| b == b';').next().unwrap_or_default();
            if size.is_empty() || !size.iter().all(u8::is_ascii_hexdigit) {
                return Err("invalid HTTP chunk size".into());
            }
            let size = usize::from_str_radix(std::str::from_utf8(size).unwrap(), 16)
                .map_err(|_| "invalid HTTP chunk size")?;
            if size > MAX_BYTES - body.len() {
                return Err("HTTP body exceeds 4 MiB".into());
            }
            if size == 0 {
                loop {
                    let trailer = read_line(reader, &mut framing_budget)?;
                    if trailer.is_empty() {
                        return Ok(body);
                    }
                    let (name, _) = header_field(&trailer)?;
                    if name == "content-length" || name == "transfer-encoding" {
                        return Err("framing headers are forbidden in HTTP trailers".into());
                    }
                }
            }
            let offset = body.len();
            body.resize(offset + size, 0);
            reader.read_exact(&mut body[offset..]).map_err(io_error)?;
            let mut crlf = [0; 2];
            reader.read_exact(&mut crlf).map_err(io_error)?;
            framing_budget = framing_budget
                .checked_sub(2)
                .ok_or("HTTP chunk framing exceeds 4 MiB")?;
            if crlf != *b"\r\n" {
                return Err("invalid HTTP chunk terminator".into());
            }
        }
    }
    if let Some(length) = content_length {
        let mut body = vec![0; length];
        reader.read_exact(&mut body).map_err(io_error)?;
        return Ok(body);
    }
    if !eof_delimited {
        // Requests without explicit framing have no body, even on keep-alive.
        return Ok(Vec::new());
    }
    let mut body = Vec::new();
    reader
        .take((MAX_BYTES + 1) as u64)
        .read_to_end(&mut body)
        .map_err(io_error)?;
    if body.len() > MAX_BYTES {
        return Err("HTTP body exceeds 4 MiB".into());
    }
    Ok(body)
}

/// Serve a single model sequentially on a loopback address, e.g. `127.0.0.1:8080`.
///
/// This diagnostic service has **no authentication or TLS**. Requests are bounded
/// and time out after 30 seconds of blocked I/O. `max_requests` counts accepted
/// connections, including malformed requests; `None` serves until an I/O error.
pub fn serve(
    backend: &mut dyn InferenceBackend,
    address: &str,
    max_requests: Option<usize>,
) -> Result<(), String> {
    let addresses = loopback_addresses(address, None)?;
    let listener = TcpListener::bind(addresses.as_slice()).map_err(io_error)?;
    serve_listener(backend, listener, max_requests)
}

/// As [`serve`], with an already-bound loopback listener (useful with port zero).
pub fn serve_listener(
    backend: &mut dyn InferenceBackend,
    listener: TcpListener,
    max_requests: Option<usize>,
) -> Result<(), String> {
    if !listener.local_addr().map_err(io_error)?.ip().is_loopback() {
        return Err("HTTP server listener must be bound to loopback only".into());
    }
    let mut count = 0;
    while max_requests.is_none_or(|limit| count < limit) {
        let (stream, peer) = listener.accept().map_err(io_error)?;
        count += 1;
        if !peer.ip().is_loopback() {
            continue;
        }
        // A malformed request, disconnected client, or write timeout must not
        // take down the service. Backend errors are returned as JSON HTTP 500s.
        let _ = serve_connection(backend, stream);
    }
    Ok(())
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Stop {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    model: Option<String>,
    #[serde(default)]
    stream: bool,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u32>,
    stop: Option<Stop>,
}

fn completion(backend: &mut dyn InferenceBackend, body: &[u8]) -> (u16, Value) {
    let request: ChatRequest = match serde_json::from_slice(body) {
        Ok(request) => request,
        Err(error) => return api_error(400, format!("malformed chat JSON: {error}")),
    };
    if request.stream {
        return api_error(400, "streaming is unsupported; set stream to false");
    }
    if request
        .model
        .as_deref()
        .is_some_and(|model| model != backend.model_id())
    {
        return api_error(
            400,
            format!(
                "unknown model ID; this server provides {:?}",
                backend.model_id()
            ),
        );
    }
    let defaults = GenerationConfig::default();
    let options = GenerationConfig {
        max_tokens: request.max_tokens.unwrap_or(defaults.max_tokens),
        temperature: request.temperature.unwrap_or(defaults.temperature),
        top_p: request.top_p.unwrap_or(defaults.top_p),
        seed: request.seed.unwrap_or(defaults.seed),
        stop: match request.stop {
            Some(Stop::One(stop)) => vec![stop],
            Some(Stop::Many(stop)) => stop,
            None => defaults.stop,
        },
    };
    if let Err(error) = validate_messages(&request.messages).and_then(|()| options.validate()) {
        return api_error(400, error);
    }
    if options.max_tokens > MAX_SERVER_TOKENS {
        return api_error(
            400,
            format!("max_tokens exceeds diagnostic server limit {MAX_SERVER_TOKENS}"),
        );
    }
    match backend.generate(&request.messages, &options) {
        Ok(generation) => {
            if generation.text.trim().is_empty() || generation.finish_reason.trim().is_empty() {
                return api_error(500, "backend returned empty content or finish_reason");
            }
            let Some(total_tokens) = generation
                .prompt_tokens
                .checked_add(generation.completion_tokens)
            else {
                return api_error(500, "backend token usage overflow");
            };
            (
                200,
                json!({
                    "id": "chatcmpl-local",
                    "object": "chat.completion",
                    "created": 0,
                    "model": backend.model_id(),
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": generation.text},
                        "finish_reason": generation.finish_reason,
                    }],
                    "usage": {
                        "prompt_tokens": generation.prompt_tokens,
                        "completion_tokens": generation.completion_tokens,
                        "total_tokens": total_tokens,
                    },
                }),
            )
        }
        Err(error) => api_error(500, error),
    }
}

fn api_error(status: u16, message: impl Into<String>) -> (u16, Value) {
    (
        status,
        json!({"error": {
            "message": message.into(),
            "type": if status >= 500 { "server_error" } else { "invalid_request_error" },
        }}),
    )
}

fn serve_connection(backend: &mut dyn InferenceBackend, stream: TcpStream) -> Result<(), String> {
    set_timeouts(&stream, SERVER_TIMEOUT)?;
    let mut reader = BufReader::new(stream);
    let response = (|| {
        let head = read_head(&mut reader)?;
        let words: Vec<_> = head.start.split_whitespace().collect();
        if words.len() != 3 || !matches!(words[2], "HTTP/1.1" | "HTTP/1.0") {
            return Err("invalid HTTP request line".into());
        }
        // Do not wait for a body from a client waiting on 100 Continue.
        if head.fields.iter().any(|(name, _)| name == "expect") {
            return Ok(api_error(417, "Expect/100-continue is unsupported"));
        }
        let body = read_body(&mut reader, &head, false)?;
        Ok(match (words[0], words[1]) {
            ("GET", "/health") => (200, json!({"status": "ok"})),
            ("GET", "/v1/models") => (
                200,
                json!({
                    "object": "list",
                    "data": [{"id": backend.model_id(), "object": "model", "created": 0, "owned_by": "local"}],
                }),
            ),
            ("POST", "/v1/chat/completions") => completion(backend, &body),
            (_, "/health" | "/v1/models" | "/v1/chat/completions") => {
                api_error(405, "method not allowed")
            }
            _ => api_error(404, "unknown HTTP route"),
        })
    })();
    let (status, body) = response.unwrap_or_else(|error: String| api_error(400, error));
    write_response(reader.get_mut(), status, &body)
}

fn write_response(stream: &mut TcpStream, mut status: u16, value: &Value) -> Result<(), String> {
    let mut body = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    if body.len() > MAX_BYTES {
        status = 500;
        body = serde_json::to_vec(&api_error(500, "HTTP response body exceeds 4 MiB").1)
            .map_err(|e| e.to_string())?;
    }
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        417 => "Expectation Failed",
        _ => "Internal Server Error",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).map_err(io_error)?;
    stream.write_all(&body).map_err(io_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::mpsc;
    use std::thread::{self, JoinHandle};

    const TIMEOUT: Duration = Duration::from_secs(2);

    fn listener() -> TcpListener {
        TcpListener::bind("127.0.0.1:0").unwrap()
    }

    fn read_request(stream: &TcpStream) -> (Head, Value) {
        set_timeouts(stream, TIMEOUT).unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let head = read_head(&mut reader).unwrap();
        let body = read_body(&mut reader, &head, false).unwrap();
        let value = if body.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&body).unwrap()
        };
        (head, value)
    }

    fn mock(
        requests: usize,
        mut handle: impl FnMut(usize, Head, Value, &mut TcpStream) + Send + 'static,
    ) -> (String, JoinHandle<()>) {
        let listener = listener();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let worker = thread::spawn(move || {
            // Bound accept too, so a failed assertion cannot strand a test worker.
            listener.set_nonblocking(true).unwrap();
            for index in 0..requests {
                let deadline = std::time::Instant::now() + Duration::from_secs(5);
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(
                                std::time::Instant::now() < deadline,
                                "mock accept timed out"
                            );
                            thread::sleep(Duration::from_millis(5));
                        }
                        Err(error) => panic!("mock accept: {error}"),
                    }
                };
                let (head, body) = read_request(&stream);
                handle(index, head, body, &mut stream);
            }
        });
        (url, worker)
    }

    fn models(stream: &mut TcpStream, ids: &[&str]) {
        let data: Vec<_> = ids.iter().map(|id| json!({"id": id})).collect();
        write_response(stream, 200, &json!({"data": data})).unwrap();
    }

    fn answer(content: Value, finish: &str) -> Value {
        json!({
            "model": "Qwen-teacher",
            "choices": [{"message": {"content": content, "reasoning_content": "PRIVATE REASONING"}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4},
        })
    }

    #[test]
    fn client_discovers_first_model_and_sends_options_roles_and_thinking_control() {
        let (url, worker) = mock(2, |index, head, request, stream| {
            if index == 0 {
                assert_eq!(head.start, "GET /v1/models HTTP/1.1");
                models(stream, &["Qwen-teacher", "other"]);
            } else {
                assert_eq!(head.start, "POST /v1/chat/completions HTTP/1.1");
                assert_eq!(request["model"], "Qwen-teacher");
                assert_eq!(
                    request["messages"],
                    json!([
                        {"role": "system", "content": "rules"},
                        {"role": "user", "content": "question"},
                        {"role": "assistant", "content": "earlier answer"},
                        {"role": "user", "content": "follow-up"},
                    ])
                );
                assert_eq!(request["max_tokens"], 19);
                assert_eq!(request["temperature"], 0.5);
                assert_eq!(request["top_p"], 0.75);
                assert_eq!(request["seed"], u32::MAX);
                assert_eq!(request["stop"], json!(["END", "停止"]));
                assert_eq!(request["stream"], false);
                assert_eq!(request["chat_template_kwargs"]["enable_thinking"], false);
                write_response(stream, 200, &answer(json!("answer 🦀"), "length")).unwrap();
            }
        });
        let mut backend = HttpBackend::connect(&format!("{url}/v1/"), None, TIMEOUT).unwrap();
        assert_eq!(backend.model_id(), "Qwen-teacher");
        let messages = vec![
            ChatMessage::system("rules"),
            ChatMessage::user("question"),
            ChatMessage {
                role: "assistant".into(),
                content: "earlier answer".into(),
            },
            ChatMessage::user("follow-up"),
        ];
        let options = GenerationConfig {
            max_tokens: 19,
            temperature: 0.5,
            top_p: 0.75,
            seed: u32::MAX,
            stop: vec!["END".into(), "停止".into()],
        };
        let generation = backend.generate(&messages, &options).unwrap();
        assert_eq!(generation.text, "answer 🦀");
        assert_eq!(generation.finish_reason, "length");
        assert_eq!(
            (generation.prompt_tokens, generation.completion_tokens),
            (9, 4)
        );
        worker.join().unwrap();
    }

    #[test]
    fn model_discovery_checks_explicit_ids_and_empty_lists() {
        for (ids, requested, expected) in [
            (vec!["one", "two"], Some("two"), Some("two")),
            (vec!["one"], Some("unknown"), None),
            (vec![], None, None),
        ] {
            let (url, worker) = mock(1, move |_, _, _, stream| models(stream, &ids));
            let result = HttpBackend::connect(&url, requested.map(str::to_owned), TIMEOUT);
            match expected {
                Some(id) => assert_eq!(result.unwrap().model_id(), id),
                None => assert!(result.is_err()),
            }
            worker.join().unwrap();
        }
    }

    #[test]
    fn client_decodes_chunked_response_and_does_not_wait_for_eof() {
        let (release_tx, release_rx) = mpsc::channel();
        let (url, worker) = mock(2, move |index, _, request, stream| {
            if index == 0 {
                models(stream, &["plain"]);
            } else {
                assert!(request.get("chat_template_kwargs").is_none());
                let body = serde_json::to_vec(&answer(json!("chunked answer"), "stop")).unwrap();
                stream.write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: Chunked\r\nConnection: keep-alive\r\n\r\n").unwrap();
                for chunk in body.chunks(7) {
                    write!(stream, "{:x};test=yes\r\n", chunk.len()).unwrap();
                    stream.write_all(chunk).unwrap();
                    stream.write_all(b"\r\n").unwrap();
                }
                stream.write_all(b"0\r\nX-Debug: trailer\r\n\r\n").unwrap();
                release_rx.recv_timeout(Duration::from_secs(4)).unwrap();
            }
        });
        let mut backend = HttpBackend::connect(&url, None, TIMEOUT).unwrap();
        let result = backend.generate(&[ChatMessage::user("hi")], &GenerationConfig::default());
        release_tx.send(()).unwrap();
        assert_eq!(result.unwrap().text, "chunked answer");
        worker.join().unwrap();
    }

    #[test]
    fn content_length_counts_bytes_and_does_not_wait_for_eof() {
        let (release_tx, release_rx) = mpsc::channel();
        let (url, worker) = mock(1, move |_, _, _, stream| {
            models(stream, &["模型"]);
            release_rx.recv_timeout(Duration::from_secs(4)).unwrap();
        });
        let result = HttpBackend::connect(&url, None, TIMEOUT);
        release_tx.send(()).unwrap();
        assert_eq!(result.unwrap().model_id(), "模型");
        worker.join().unwrap();
    }

    #[test]
    fn client_reports_http_errors_and_bad_answers() {
        let (url, worker) = mock(1, |_, _, _, stream| {
            write_response(
                stream,
                503,
                &json!({"error": {"message": "model unavailable"}}),
            )
            .unwrap();
        });
        let error = HttpBackend::connect(&url, None, TIMEOUT).err().unwrap();
        assert!(error.contains("HTTP 503") && error.contains("model unavailable"));
        worker.join().unwrap();

        for body in [
            answer(Value::Null, "stop"),
            answer(json!("  "), "stop"),
            json!({"choices": []}),
            json!({"choices": [{"message": {"reasoning_content": "not an answer"}, "finish_reason": "stop"}]}),
        ] {
            let (url, worker) = mock(2, move |index, _, _, stream| {
                if index == 0 {
                    models(stream, &["one"]);
                } else {
                    write_response(stream, 200, &body).unwrap();
                }
            });
            let mut backend = HttpBackend::connect(&url, None, TIMEOUT).unwrap();
            assert!(
                backend
                    .generate(&[ChatMessage::user("hi")], &GenerationConfig::default())
                    .is_err()
            );
            worker.join().unwrap();
        }
    }

    #[test]
    fn rejects_remote_urls_tls_and_invalid_configuration_without_connecting() {
        for url in [
            "https://127.0.0.1",
            "http://example.com",
            "http://0.0.0.0:1234",
            "http://192.168.0.1",
            "http://[::]",
            "http://[::ffff:127.0.0.1]",
            "http://localhost@evil.test",
            "http://localhost:80\r\nInjected: yes",
            "http://localhost?query",
            "http://localhost/#fragment",
            "http://localhost:bad",
        ] {
            assert!(Endpoint::parse(url).is_err(), "accepted {url:?}");
        }
        assert!(Endpoint::parse("http://localhost:1234").is_ok());
        assert!(Endpoint::parse("http://[::1]:1234/v1").is_ok());
        assert_eq!(
            Endpoint::parse("http://127.0.0.1/prefix/v1/")
                .unwrap()
                .prefix,
            "/prefix/v1"
        );
        assert!(HttpBackend::connect("http://127.0.0.1:1", None, Duration::ZERO).is_err());
        let mut backend = HttpBackend {
            endpoint: Endpoint::parse("http://127.0.0.1:1").unwrap(),
            model: "test".into(),
            timeout: TIMEOUT,
        };
        assert!(
            backend
                .generate(&[], &GenerationConfig::default())
                .unwrap_err()
                .contains("message")
        );
        let bad_role = ChatMessage {
            role: "tool".into(),
            content: "x".into(),
        };
        assert!(
            backend
                .generate(&[bad_role], &GenerationConfig::default())
                .unwrap_err()
                .contains("role")
        );
        let options = GenerationConfig {
            max_tokens: 0,
            ..GenerationConfig::default()
        };
        assert!(
            backend
                .generate(&[ChatMessage::user("x")], &options)
                .unwrap_err()
                .contains("max_tokens")
        );
    }

    fn decoded(raw: &[u8], eof: bool) -> Result<Vec<u8>, String> {
        let mut reader = Cursor::new(raw);
        let head = read_head(&mut reader)?;
        read_body(&mut reader, &head, eof)
    }

    #[test]
    fn framing_rejects_truncation_ambiguity_and_oversized_data() {
        for raw in [
            "HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\nshort",
            "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n",
            "HTTP/1.1 200 OK\r\nContent-Length: +1\r\n\r\nx",
            "HTTP/1.1 200 OK\r\nContent-Length: 4194305\r\n\r\n",
            "HTTP/1.1 200 OK\r\nContent-Length: 1\r\nTransfer-Encoding: chunked\r\n\r\n",
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip\r\n\r\n",
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nZ\r\n",
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n400001\r\n",
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nx",
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n1\r\nxNO",
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\nContent-Length: 0\r\n\r\n",
            "HTTP/1.1 200 OK\nContent-Length: 0\n\n",
            "HTTP/1.1 200 OK\r\n folded: nope\r\n\r\n",
        ] {
            assert!(decoded(raw.as_bytes(), true).is_err(), "accepted {raw:?}");
        }
        assert_eq!(
            decoded(b"HTTP/1.0 200 OK\r\n\r\neof body", true).unwrap(),
            b"eof body"
        );
        assert_eq!(decoded(b"POST / HTTP/1.1\r\n\r\n", false).unwrap(), b"");
        let mut oversized = b"HTTP/1.1 200 OK\r\nX-Big: ".to_vec();
        oversized.extend(vec![b'x'; MAX_BYTES]);
        assert!(decoded(&oversized, true).is_err());
        let mut oversized = b"HTTP/1.1 200 OK\r\n\r\n".to_vec();
        oversized.extend(vec![b'x'; MAX_BYTES + 1]);
        assert!(decoded(&oversized, true).is_err());
    }

    #[derive(Default)]
    struct FakeBackend {
        calls: Vec<(Vec<ChatMessage>, GenerationConfig)>,
    }

    impl InferenceBackend for FakeBackend {
        fn model_id(&self) -> &str {
            "fake-model"
        }
        fn generate(
            &mut self,
            messages: &[ChatMessage],
            options: &GenerationConfig,
        ) -> Result<Generation, String> {
            self.calls.push((messages.to_vec(), options.clone()));
            Ok(Generation {
                text: "local answer 🦀".into(),
                model: "fake-model".into(),
                prompt_tokens: 7,
                completion_tokens: 3,
                finish_reason: "stop".into(),
            })
        }
    }

    fn exchange(address: SocketAddr, method: &str, route: &str, body: &[u8]) -> (u16, Value) {
        let mut request = format!(
            "{method} {route} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )
        .into_bytes();
        request.extend_from_slice(body);
        raw_exchange(address, &request)
    }

    fn raw_exchange(address: SocketAddr, request: &[u8]) -> (u16, Value) {
        let mut stream = TcpStream::connect_timeout(&address, TIMEOUT).unwrap();
        set_timeouts(&stream, TIMEOUT).unwrap();
        stream.write_all(request).unwrap();
        let mut reader = BufReader::new(stream);
        let head = read_head(&mut reader).unwrap();
        let status = head
            .start
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        let body = read_body(&mut reader, &head, true).unwrap();
        (status, serde_json::from_slice(&body).unwrap())
    }

    #[test]
    fn server_routes_recovery_validation_and_options() {
        let listener = listener();
        let address = listener.local_addr().unwrap();
        let invalid = vec![
            b"not json".to_vec(),
            serde_json::to_vec(&json!({"messages": [{"role": "user", "content": "x"}], "stream": true})).unwrap(),
            serde_json::to_vec(&json!({"messages": [{"role": "user", "content": "x"}], "model": "wrong"})).unwrap(),
            serde_json::to_vec(&json!({"messages": []})).unwrap(),
            serde_json::to_vec(&json!({"messages": [{"role": "tool", "content": "x"}]})).unwrap(),
            serde_json::to_vec(&json!({"messages": [{"role": "user", "content": " "}]})).unwrap(),
            serde_json::to_vec(&json!({"messages": [{"role": "user", "content": "x"}], "max_tokens": 0})).unwrap(),
            serde_json::to_vec(&json!({"messages": [{"role": "user", "content": "x"}], "max_tokens": MAX_SERVER_TOKENS + 1})).unwrap(),
            serde_json::to_vec(&json!({"messages": [{"role": "user", "content": "x"}], "temperature": -1})).unwrap(),
            serde_json::to_vec(&json!({"messages": [{"role": "user", "content": "x"}], "top_p": 0})).unwrap(),
            serde_json::to_vec(&json!({"messages": [{"role": "user", "content": "x"}], "seed": -1})).unwrap(),
            serde_json::to_vec(&json!({"messages": [{"role": "user", "content": "x"}], "stop": [""]})).unwrap(),
        ];
        let count = invalid.len() + 5;
        let worker = thread::spawn(move || {
            let mut backend = FakeBackend::default();
            serve_listener(&mut backend, listener, Some(count)).unwrap();
            backend
        });
        for body in invalid {
            let (status, response) = exchange(address, "POST", "/v1/chat/completions", &body);
            assert_eq!(status, 400, "{response}");
            assert!(response["error"]["message"].is_string());
        }
        assert_eq!(exchange(address, "GET", "/health", b"").0, 200);
        let (status, models) = exchange(address, "GET", "/v1/models", b"");
        assert_eq!(status, 200);
        assert_eq!(models["data"][0]["id"], "fake-model");
        assert_eq!(exchange(address, "GET", "/missing", b"").0, 404);
        assert_eq!(exchange(address, "GET", "/v1/chat/completions", b"").0, 405);
        let request = json!({
            "model": "fake-model", "messages": [{"role": "system", "content": "rules"}, {"role": "user", "content": "hi"}],
            "max_tokens": 17, "temperature": 0.5, "top_p": 0.75, "seed": 99,
            "stop": "END", "stream": false, "chat_template_kwargs": {"enable_thinking": false},
        });
        let (status, response) = exchange(
            address,
            "POST",
            "/v1/chat/completions",
            &serde_json::to_vec(&request).unwrap(),
        );
        assert_eq!(status, 200);
        assert_eq!(response["choices"][0]["message"]["role"], "assistant");
        assert_eq!(
            response["choices"][0]["message"]["content"],
            "local answer 🦀"
        );
        assert_eq!(response["usage"]["total_tokens"], 10);
        let backend = worker.join().unwrap();
        assert_eq!(backend.calls.len(), 1);
        let (messages, options) = &backend.calls[0];
        assert_eq!(messages[0], ChatMessage::system("rules"));
        assert_eq!(options.max_tokens, 17);
        assert_eq!(
            (options.temperature, options.top_p, options.seed),
            (0.5, 0.75, 99)
        );
        assert_eq!(options.stop, ["END"]);
    }

    #[test]
    fn client_read_timeout_is_enforced() {
        let (release_tx, release_rx) = mpsc::channel();
        let (url, worker) = mock(1, move |_, _, _, _| {
            release_rx.recv_timeout(Duration::from_secs(4)).unwrap();
        });
        let result = HttpBackend::connect(&url, None, Duration::from_millis(100));
        release_tx.send(()).unwrap();
        let error = result.err().expect("unresponsive server must time out");
        assert!(error.contains("HTTP I/O error"), "{error}");
        worker.join().unwrap();
    }

    #[test]
    fn client_handles_eof_delimited_discovery_and_rejects_bad_chat_transport() {
        let (url, worker) = mock(1, |_, _, _, stream| {
            stream
                .write_all(b"HTTP/1.0 200 OK\r\n\r\n{\"data\":[{\"id\":\"one\"}]}")
                .unwrap();
        });
        assert_eq!(
            HttpBackend::connect(&url, None, TIMEOUT)
                .unwrap()
                .model_id(),
            "one"
        );
        worker.join().unwrap();

        for (raw, expected) in [
            (
                "HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{}",
                "HTTP I/O error",
            ),
            (
                "HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\n{",
                "invalid chat completion response",
            ),
            (
                "HTTP/1.1 429 Too Many Requests\r\nContent-Length: 4\r\n\r\nslow",
                "HTTP 429",
            ),
        ] {
            let (url, worker) = mock(2, move |index, _, _, stream| {
                if index == 0 {
                    models(stream, &["one"]);
                } else {
                    stream.write_all(raw.as_bytes()).unwrap();
                }
            });
            let mut client = HttpBackend::connect(&url, None, TIMEOUT).unwrap();
            let error = client
                .generate(&[ChatMessage::user("hi")], &GenerationConfig::default())
                .unwrap_err();
            assert!(error.contains(expected), "{error}");
            worker.join().unwrap();
        }
    }

    #[test]
    fn server_accepts_chunked_requests_and_recovers_from_bad_http() {
        let listener = listener();
        let address = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            let mut backend = FakeBackend::default();
            serve_listener(&mut backend, listener, Some(4)).unwrap();
            backend
        });
        assert_eq!(
            raw_exchange(
                address,
                b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: -1\r\n\r\n"
            )
            .0,
            400
        );
        assert_eq!(raw_exchange(address, b"POST /v1/chat/completions HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: 100\r\n\r\n").0, 417);
        let body = serde_json::to_vec(&json!({
            "messages": [{"role": "user", "content": "hello"}], "seed": 123,
            "stop": ["ONE", "TWO"],
        }))
        .unwrap();
        let mut request = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        for chunk in body.chunks(5) {
            write!(request, "{:x}\r\n", chunk.len()).unwrap();
            request.extend_from_slice(chunk);
            request.extend_from_slice(b"\r\n");
        }
        request.extend_from_slice(b"0\r\n\r\n");
        assert_eq!(raw_exchange(address, &request).0, 200);
        // A request with no length and a live connection has an empty body.
        assert_eq!(
            raw_exchange(address, b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n").0,
            200
        );
        let backend = worker.join().unwrap();
        assert_eq!(backend.calls.len(), 1);
        assert_eq!(backend.calls[0].1.seed, 123);
        assert_eq!(backend.calls[0].1.stop, ["ONE", "TWO"]);
    }

    #[test]
    fn backend_errors_are_openai_shaped_http_errors() {
        struct FailingBackend;
        impl InferenceBackend for FailingBackend {
            fn model_id(&self) -> &str {
                "broken"
            }
            fn generate(
                &mut self,
                _: &[ChatMessage],
                _: &GenerationConfig,
            ) -> Result<Generation, String> {
                Err("diagnostic backend failed".into())
            }
        }
        let listener = listener();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let worker = thread::spawn(move || {
            serve_listener(&mut FailingBackend, listener, Some(2)).unwrap();
        });
        let mut client = HttpBackend::connect(&url, None, TIMEOUT).unwrap();
        let error = client
            .generate(&[ChatMessage::user("hi")], &GenerationConfig::default())
            .unwrap_err();
        assert!(
            error.contains("HTTP 500") && error.contains("diagnostic backend failed"),
            "{error}"
        );
        worker.join().unwrap();
    }

    #[test]
    fn client_and_server_round_trip_and_zero_request_limit() {
        let listener = listener();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let worker = thread::spawn(move || {
            serve_listener(&mut FakeBackend::default(), listener, Some(2)).unwrap();
        });
        let mut client = HttpBackend::connect(&url, Some("fake-model".into()), TIMEOUT).unwrap();
        let result = client
            .generate(&[ChatMessage::user("hi")], &GenerationConfig::default())
            .unwrap();
        assert_eq!(result.text, "local answer 🦀");
        assert_eq!(result.finish_reason, "stop");
        worker.join().unwrap();
        serve_listener(&mut FakeBackend::default(), self::listener(), Some(0)).unwrap();
        assert!(serve(&mut FakeBackend::default(), "0.0.0.0:0", Some(0)).is_err());
        let wildcard = TcpListener::bind("0.0.0.0:0").unwrap();
        assert!(serve_listener(&mut FakeBackend::default(), wildcard, Some(0)).is_err());
    }
}
