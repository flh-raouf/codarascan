# Framed worker protocol version 1

`codarascan _worker` is a private persistent local process for backend language
adapters. It reads standard input and writes standard output. It opens no
socket, HTTP server, or external connection. Warnings use standard error.

The request pool defaults to one worker. Set `codarascan _worker --workers 4`
or `python -m codarascan.worker --workers 4` to process up to four requests
concurrently. Python callers can use `WorkerServer(reader, writer, workers=4)`.
The count must be a positive integer; it is never inferred from CPU count.
This controls concurrent requests. The separate `workers` parameter for
`scan_document` and `iter_document` controls page analysis within each request
and still defaults to 1 (with explicit `"auto"` supported).

Each message is four bytes containing an unsigned big-endian payload length,
followed by exactly that many UTF-8 JSON bytes. JSON may contain newlines and
may be larger than a line. The worker applies no additional public frame-size
limit.

Requests have this envelope:

```json
{"protocol":1,"id":"request-7","operation":"scan_image","params":{}}
```

Responses repeat `protocol` and `id`, include `ok`, and contain either `result`
or a structured `error` with `code`, `message`, and safe `context`. Complete
responses use `event="complete"`. `iter_document` emits zero or more
`event="page"` responses followed by one complete document summary. Different
active request IDs may complete out of order.

Operations are `capabilities`, `create_scanner`, `warm`, `scan_image`,
`scan_document`, `iter_document`, `release_scanner`, `cancel`, and `shutdown`.
Image/document input is a path string, `{"path":"..."}`, or
`{"base64":"..."}`. Scanner creation accepts the public mode/symbols/formats/
decode object and returns a reusable `scanner_id`.

Cancellation stops document scheduling at a page boundary and requests safe
future cancellation. Native work already executing may finish before the
cancelled response. Shutdown is graceful and waits for active requests.

Protocol mismatch, malformed frames, duplicate active IDs, unknown operations,
unknown scanner IDs, and invalid Base64 return `protocol_error`; arbitrary
Python tracebacks are never returned by default.
