use crate::channels::{Channel, InboundMessage};
use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Minimal WebSocket frame helpers (RFC 6455 basics)
// ---------------------------------------------------------------------------

fn parse_websocket_key(request: &str) -> Option<String> {
    for line in request.lines() {
        if line.to_lowercase().starts_with("sec-websocket-key:") {
            return Some(line.split(':').nth(1)?.trim().to_string());
        }
    }
    None
}

fn compute_accept_key(key: &str) -> String {
    // SHA-1 of key + magic GUID, base64-encoded
    let magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    let combined = format!("{}{}", key, magic);
    let hash = sha1_digest(combined.as_bytes());
    base64_encode(&hash)
}

/// Minimal SHA-1 implementation (RFC 3174) — just enough for WebSocket handshake.
fn sha1_digest(data: &[u8]) -> [u8; 20] {
    let mut h0: u32 = 0x67452301;
    let mut h1: u32 = 0xEFCDAB89;
    let mut h2: u32 = 0x98BADCFE;
    let mut h3: u32 = 0x10325476;
    let mut h4: u32 = 0xC3D2E1F0;

    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h0, h1, h2, h3, h4);
        for (i, wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A827999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1u32),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDCu32),
                _ => (b ^ c ^ d, 0xCA62C1D6u32),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }
        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut result = [0u8; 20];
    result[0..4].copy_from_slice(&h0.to_be_bytes());
    result[4..8].copy_from_slice(&h1.to_be_bytes());
    result[8..12].copy_from_slice(&h2.to_be_bytes());
    result[12..16].copy_from_slice(&h3.to_be_bytes());
    result[16..20].copy_from_slice(&h4.to_be_bytes());
    result
}

fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

/// Read a single WebSocket text frame. Returns None on close/error.
fn read_ws_frame(stream: &mut TcpStream) -> Option<String> {
    let mut header = [0u8; 2];
    stream.read_exact(&mut header).ok()?;

    let opcode = header[0] & 0x0F;
    if opcode == 0x08 {
        return None; // close frame
    }

    let masked = (header[1] & 0x80) != 0;
    let mut payload_len = (header[1] & 0x7F) as u64;

    if payload_len == 126 {
        let mut ext = [0u8; 2];
        stream.read_exact(&mut ext).ok()?;
        payload_len = u16::from_be_bytes(ext) as u64;
    } else if payload_len == 127 {
        let mut ext = [0u8; 8];
        stream.read_exact(&mut ext).ok()?;
        payload_len = u64::from_be_bytes(ext);
    }

    let mask_key = if masked {
        let mut key = [0u8; 4];
        stream.read_exact(&mut key).ok()?;
        Some(key)
    } else {
        None
    };

    let mut payload = vec![0u8; payload_len as usize];
    stream.read_exact(&mut payload).ok()?;

    if let Some(key) = mask_key {
        for (i, byte) in payload.iter_mut().enumerate() {
            *byte ^= key[i % 4];
        }
    }

    if opcode == 0x01 {
        String::from_utf8(payload).ok()
    } else {
        None
    }
}

/// Write a WebSocket text frame.
fn write_ws_frame(stream: &mut TcpStream, text: &str) -> Result<(), String> {
    let payload = text.as_bytes();
    let mut frame = Vec::new();

    frame.push(0x81); // FIN + text opcode
    if payload.len() < 126 {
        frame.push(payload.len() as u8);
    } else if payload.len() < 65536 {
        frame.push(126);
        frame.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    } else {
        frame.push(127);
        frame.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    }
    frame.extend_from_slice(payload);

    stream.write_all(&frame).map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// WebSocket Channel
// ---------------------------------------------------------------------------

const MAX_INCOMING_QUEUE: usize = 1024;

pub struct WebSocketChannel {
    incoming: Arc<Mutex<VecDeque<InboundMessage>>>,
    connections: Arc<Mutex<HashMap<String, TcpStream>>>,
}

impl Default for WebSocketChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl WebSocketChannel {
    pub fn new() -> Self {
        Self {
            incoming: Arc::new(Mutex::new(VecDeque::new())),
            connections: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl Channel for WebSocketChannel {
    fn name(&self) -> &str {
        "websocket"
    }

    fn recv(&self) -> Option<InboundMessage> {
        self.incoming.lock().unwrap().pop_front()
    }

    fn send(&self, peer_id: &str, text: &str) -> Result<(), String> {
        let mut conns = self.connections.lock().unwrap();
        if let Some(stream) = conns.get_mut(peer_id) {
            write_ws_frame(stream, text)
        } else {
            Err(format!("Peer '{}' not connected", peer_id))
        }
    }
}

// ---------------------------------------------------------------------------
// Gateway: WebSocket server with binding table
// ---------------------------------------------------------------------------

/// 5-tier binding: (channel, peer_id) → agent_name
pub struct Gateway {
    addr: String,
    ws_channel: Arc<WebSocketChannel>,
    bindings: Arc<Mutex<HashMap<(String, String), String>>>,
}

impl Gateway {
    pub fn new(addr: &str) -> Self {
        Self {
            addr: addr.to_string(),
            ws_channel: Arc::new(WebSocketChannel::new()),
            bindings: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get a reference to the WebSocket channel for registration with ChannelManager.
    pub fn ws_channel(&self) -> Arc<WebSocketChannel> {
        Arc::clone(&self.ws_channel)
    }

    /// Bind a (channel, peer) to an agent.
    pub fn bind(&self, channel: &str, peer_id: &str, agent: &str) {
        self.bindings.lock().unwrap().insert(
            (channel.to_string(), peer_id.to_string()),
            agent.to_string(),
        );
    }

    /// Look up which agent handles a (channel, peer).
    pub fn resolve(&self, channel: &str, peer_id: &str) -> Option<String> {
        self.bindings
            .lock()
            .unwrap()
            .get(&(channel.to_string(), peer_id.to_string()))
            .cloned()
    }

    /// Start the WebSocket listener in a background thread.
    pub fn start(&self) {
        let addr = self.addr.clone();
        let incoming = Arc::clone(&self.ws_channel.incoming);
        let connections = Arc::clone(&self.ws_channel.connections);

        std::thread::spawn(move || {
            let listener = match TcpListener::bind(&addr) {
                Ok(l) => {
                    eprintln!("[gateway] WebSocket server listening on {}", addr);
                    l
                }
                Err(e) => {
                    eprintln!("[gateway] Failed to bind {}: {}", addr, e);
                    return;
                }
            };

            for stream in listener.incoming() {
                let stream = match stream {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                let incoming = Arc::clone(&incoming);
                let connections = Arc::clone(&connections);

                std::thread::spawn(move || {
                    handle_ws_connection(stream, incoming, connections);
                });
            }
        });
    }
}

fn handle_ws_connection(
    mut stream: TcpStream,
    incoming: Arc<Mutex<VecDeque<InboundMessage>>>,
    connections: Arc<Mutex<HashMap<String, TcpStream>>>,
) {
    // Read HTTP upgrade request
    let mut buf = [0u8; 4096];
    let n = match stream.read(&mut buf) {
        Ok(n) => n,
        Err(_) => return,
    };
    let request = String::from_utf8_lossy(&buf[..n]).to_string();

    // WebSocket handshake
    let ws_key = match parse_websocket_key(&request) {
        Some(k) => k,
        None => return, // Not a WebSocket request
    };
    let accept = compute_accept_key(&ws_key);
    let response = format!(
        "HTTP/1.1 101 Switching Protocols\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Accept: {}\r\n\r\n",
        accept
    );
    if stream.write_all(response.as_bytes()).is_err() {
        return;
    }

    let peer_id = format!(
        "ws_{}",
        stream
            .peer_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|_| "unknown".into())
    );

    // Register connection
    if let Ok(cloned) = stream.try_clone() {
        connections.lock().unwrap().insert(peer_id.clone(), cloned);
    }

    // Read loop
    loop {
        match read_ws_frame(&mut stream) {
            Some(text) => {
                let msg = InboundMessage {
                    text,
                    sender_id: peer_id.clone(),
                    channel: "websocket".to_string(),
                    peer_id: peer_id.clone(),
                    is_group: false,
                    media: Vec::new(),
                    raw: serde_json::Value::Null,
                };
                let mut queue = incoming.lock().unwrap();
                if queue.len() < MAX_INCOMING_QUEUE {
                    queue.push_back(msg);
                } else {
                    eprintln!("[gateway] Dropping message from {}: queue full", peer_id);
                }
            }
            None => {
                // Connection closed
                connections.lock().unwrap().remove(&peer_id);
                break;
            }
        }
    }
}
