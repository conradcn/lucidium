/**
 * Init-script payload that replaces window.WebSocket with a mock.
 *
 * Runs as a plain script string injected via Playwright's
 * addInitScript on every navigation. Must remain syntactically pure
 * JavaScript — no TypeScript syntax — because it isn't compiled.
 *
 * Tests configure ``window.__lucidium.handlers`` to map message types
 * to factories that return a list of s2c/* envelopes; the mock
 * delivers them asynchronously through onmessage.
 */

(function () {
  if (window.__lucidium) return;

  function MockWebSocket(url) {
    this.url = url;
    this.readyState = 0;
    this.onopen = null;
    this.onmessage = null;
    this.onerror = null;
    this.onclose = null;
    window.__lucidium.sockets.push(this);
    // Arrow callback: captures the instance lexically, so there is no
    // need to alias ``this`` into a ``self`` local.
    setTimeout(() => {
      this.readyState = 1;
      if (this.onopen) this.onopen({});
    }, 5);
  }

  MockWebSocket.CONNECTING = 0;
  MockWebSocket.OPEN = 1;
  MockWebSocket.CLOSING = 2;
  MockWebSocket.CLOSED = 3;

  MockWebSocket.prototype.addEventListener = function (type, handler) {
    if (type === "open") this.onopen = handler;
    else if (type === "message") this.onmessage = handler;
    else if (type === "close") this.onclose = handler;
    else if (type === "error") this.onerror = handler;
  };

  MockWebSocket.prototype.removeEventListener = function () {
    /* noop */
  };

  MockWebSocket.prototype.send = function (data) {
    var envelope;
    try {
      envelope = JSON.parse(data);
    } catch (_e) {
      return;
    }
    window.__lucidium.sent.push(envelope);
    var handler = window.__lucidium.handlers[envelope.type];
    if (!handler) return;
    // Arrow callbacks throughout: ``this`` stays the socket that
    // ``send`` was invoked on, with no ``self`` alias.
    Promise.resolve(handler(envelope.payload)).then((replies) => {
      (replies || []).forEach((reply) => {
        // Per-reply delay overrides the default 5ms cadence — used by
        // tests to simulate slow LLM-driven backend responses.
        var delay = typeof reply.__delayMs === "number" ? reply.__delayMs : 5;
        // Strip the delay marker so the renderer never sees it.
        var clean = Object.assign({}, reply);
        delete clean.__delayMs;
        setTimeout(() => {
          if (this.onmessage) this.onmessage({ data: JSON.stringify(clean) });
        }, delay);
      });
    });
  };

  MockWebSocket.prototype.close = function () {
    this.readyState = 3;
    if (this.onclose) this.onclose({});
  };

  window.__lucidium = {
    sent: [],
    handlers: {
      "c2s/hello": function () {
        return [
          {
            type: "s2c/hello",
            payload: { protocol_version: 1, has_save: false },
          },
        ];
      },
    },
    sockets: [],
    pushFromServer: function (envelope) {
      var data = JSON.stringify(envelope);
      window.__lucidium.sockets.forEach(function (socket) {
        if (socket.readyState === 1 && socket.onmessage) {
          socket.onmessage({ data: data });
        }
      });
    },
  };
  window.WebSocket = MockWebSocket;
})();
