/*!
 * ONNX Runtime Web v1.15.1
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
var _scriptDir,
  e =
    ((_scriptDir =
      "undefined" != typeof document && document.currentScript
        ? document.currentScript.src
        : void 0),
    "undefined" != typeof __filename && (_scriptDir = _scriptDir || __filename),
    function (e) {
      function n() {
        return R.buffer != E && q(R.buffer), W;
      }
      function t() {
        return R.buffer != E && q(R.buffer), H;
      }
      function r() {
        return R.buffer != E && q(R.buffer), F;
      }
      function a() {
        return R.buffer != E && q(R.buffer), N;
      }
      function o() {
        return R.buffer != E && q(R.buffer), P;
      }
      var i, u, s;
      (e = e || {}),
        i || (i = void 0 !== e ? e : {}),
        (i.ready = new Promise(function (e, n) {
          (u = e), (s = n);
        })),
        (i.jsepInit = function (e, n, t, r, a, o, u, s) {
          (i.Qb = e),
            (i.zb = n),
            (i.Bb = t),
            (i.gb = r),
            (i.Ab = a),
            (i.Da = o),
            (i.Cb = u),
            (i.Db = s);
        });
      var f,
        c,
        l,
        p,
        d,
        m,
        h = Object.assign({}, i),
        _ = "./this.program",
        b = (e, n) => {
          throw n;
        },
        g = "object" == typeof window,
        y = "function" == typeof importScripts,
        v =
          "object" == typeof process &&
          "object" == typeof process.versions &&
          "string" == typeof process.versions.node,
        w = i.ENVIRONMENT_IS_PTHREAD || !1,
        D = "";
      function T(e) {
        return i.locateFile ? i.locateFile(e, D) : D + e;
      }
      if (v) {
        let e;
        (D = y ? require("path").dirname(D) + "/" : __dirname + "/"),
          (m = () => {
            d || ((p = require("fs")), (d = require("path")));
          }),
          (f = function (e, n) {
            return (
              m(), (e = d.normalize(e)), p.readFileSync(e, n ? void 0 : "utf8")
            );
          }),
          (l = (e) => ((e = f(e, !0)).buffer || (e = new Uint8Array(e)), e)),
          (c = (e, n, t) => {
            m(),
              (e = d.normalize(e)),
              p.readFile(e, function (e, r) {
                e ? t(e) : n(r.buffer);
              });
          }),
          1 < process.argv.length && (_ = process.argv[1].replace(/\\/g, "/")),
          process.argv.slice(2),
          process.on("uncaughtException", function (e) {
            if (!(e instanceof ae)) throw e;
          }),
          process.on("unhandledRejection", function (e) {
            throw e;
          }),
          (b = (e, n) => {
            if (x) throw ((process.exitCode = e), n);
            n instanceof ae || M("exiting due to exception: " + n),
              process.exit(e);
          }),
          (i.inspect = function () {
            return "[Emscripten Module object]";
          });
        try {
          e = require("worker_threads");
        } catch (e) {
          throw (
            (console.error(
              'The "worker_threads" module is not supported in this node.js build - perhaps a newer version is needed?'
            ),
            e)
          );
        }
        global.Worker = e.Worker;
      } else
        (g || y) &&
          (y
            ? (D = self.location.href)
            : "undefined" != typeof document &&
              document.currentScript &&
              (D = document.currentScript.src),
          _scriptDir && (D = _scriptDir),
          (D =
            0 !== D.indexOf("blob:")
              ? D.substr(0, D.replace(/[?#].*/, "").lastIndexOf("/") + 1)
              : ""),
          v ||
            ((f = (e) => {
              var n = new XMLHttpRequest();
              return n.open("GET", e, !1), n.send(null), n.responseText;
            }),
            y &&
              (l = (e) => {
                var n = new XMLHttpRequest();
                return (
                  n.open("GET", e, !1),
                  (n.responseType = "arraybuffer"),
                  n.send(null),
                  new Uint8Array(n.response)
                );
              }),
            (c = (e, n, t) => {
              var r = new XMLHttpRequest();
              r.open("GET", e, !0),
                (r.responseType = "arraybuffer"),
                (r.onload = () => {
                  200 == r.status || (0 == r.status && r.response)
                    ? n(r.response)
                    : t();
                }),
                (r.onerror = t),
                r.send(null);
            })));
      v &&
        "undefined" == typeof performance &&
        (global.performance = require("perf_hooks").performance);
      var O = console.log.bind(console),
        A = console.warn.bind(console);
      v &&
        (m(),
        (O = (e) => p.writeSync(1, e + "\n")),
        (A = (e) => p.writeSync(2, e + "\n")));
      var S,
        C = i.print || O,
        M = i.printErr || A;
      Object.assign(i, h),
        (h = null),
        i.thisProgram && (_ = i.thisProgram),
        i.quit && (b = i.quit),
        i.wasmBinary && (S = i.wasmBinary);
      var x = i.noExitRuntime || !0;
      "object" != typeof WebAssembly && ee("no native wasm support detected");
      var R,
        k,
        E,
        W,
        H,
        F,
        N,
        P,
        U = !1,
        I =
          "undefined" != typeof TextDecoder ? new TextDecoder("utf8") : void 0;
      function Y(e, n, t) {
        var r = (n >>>= 0) + t;
        for (t = n; e[t] && !(t >= r); ) ++t;
        if (16 < t - n && e.buffer && I)
          return I.decode(
            e.buffer instanceof SharedArrayBuffer
              ? e.slice(n, t)
              : e.subarray(n, t)
          );
        for (r = ""; n < t; ) {
          var a = e[n++];
          if (128 & a) {
            var o = 63 & e[n++];
            if (192 == (224 & a)) r += String.fromCharCode(((31 & a) << 6) | o);
            else {
              var i = 63 & e[n++];
              65536 >
              (a =
                224 == (240 & a)
                  ? ((15 & a) << 12) | (o << 6) | i
                  : ((7 & a) << 18) | (o << 12) | (i << 6) | (63 & e[n++]))
                ? (r += String.fromCharCode(a))
                : ((a -= 65536),
                  (r += String.fromCharCode(
                    55296 | (a >> 10),
                    56320 | (1023 & a)
                  )));
            }
          } else r += String.fromCharCode(a);
        }
        return r;
      }
      function j(e, n) {
        return (e >>>= 0) ? Y(t(), e, n) : "";
      }
      function B(e, n, t, r) {
        if (!(0 < r)) return 0;
        var a = (t >>>= 0);
        r = t + r - 1;
        for (var o = 0; o < e.length; ++o) {
          var i = e.charCodeAt(o);
          if (
            (55296 <= i &&
              57343 >= i &&
              (i = (65536 + ((1023 & i) << 10)) | (1023 & e.charCodeAt(++o))),
            127 >= i)
          ) {
            if (t >= r) break;
            n[t++ >>> 0] = i;
          } else {
            if (2047 >= i) {
              if (t + 1 >= r) break;
              n[t++ >>> 0] = 192 | (i >> 6);
            } else {
              if (65535 >= i) {
                if (t + 2 >= r) break;
                n[t++ >>> 0] = 224 | (i >> 12);
              } else {
                if (t + 3 >= r) break;
                (n[t++ >>> 0] = 240 | (i >> 18)),
                  (n[t++ >>> 0] = 128 | ((i >> 12) & 63));
              }
              n[t++ >>> 0] = 128 | ((i >> 6) & 63);
            }
            n[t++ >>> 0] = 128 | (63 & i);
          }
        }
        return (n[t >>> 0] = 0), t - a;
      }
      function G(e) {
        for (var n = 0, t = 0; t < e.length; ++t) {
          var r = e.charCodeAt(t);
          127 >= r
            ? n++
            : 2047 >= r
            ? (n += 2)
            : 55296 <= r && 57343 >= r
            ? ((n += 4), ++t)
            : (n += 3);
        }
        return n;
      }
      function q(e) {
        (E = e),
          (i.HEAP8 = W = new Int8Array(e)),
          (i.HEAP16 = new Int16Array(e)),
          (i.HEAP32 = F = new Int32Array(e)),
          (i.HEAPU8 = H = new Uint8Array(e)),
          (i.HEAPU16 = new Uint16Array(e)),
          (i.HEAPU32 = N = new Uint32Array(e)),
          (i.HEAPF32 = new Float32Array(e)),
          (i.HEAPF64 = P = new Float64Array(e));
      }
      w && (E = i.buffer);
      var z = i.INITIAL_MEMORY || 16777216;
      if (w) (R = i.wasmMemory), (E = i.buffer);
      else if (i.wasmMemory) R = i.wasmMemory;
      else if (
        !(
          (R = new WebAssembly.Memory({
            initial: z / 65536,
            maximum: 65536,
            shared: !0,
          })).buffer instanceof SharedArrayBuffer
        )
      )
        throw (
          (M(
            "requested a shared WebAssembly.Memory but the returned buffer is not a SharedArrayBuffer, indicating that while the browser has SharedArrayBuffer it does not have WebAssembly threads support - you may need to set a flag"
          ),
          v &&
            console.log(
              "(on node you may need: --experimental-wasm-threads --experimental-wasm-bulk-memory and also use a recent version)"
            ),
          Error("bad memory"))
        );
      R && (E = R.buffer), (z = E.byteLength), q(E);
      var L = [],
        V = [],
        X = [];
      function J() {
        var e = i.preRun.shift();
        L.unshift(e);
      }
      var Z,
        $ = 0,
        Q = null,
        K = null;
      function ee(e) {
        throw (
          (w
            ? postMessage({ cmd: "onAbort", arg: e })
            : i.onAbort && i.onAbort(e),
          M((e = "Aborted(" + e + ")")),
          (U = !0),
          (e = new WebAssembly.RuntimeError(
            e + ". Build with -sASSERTIONS for more info."
          )),
          s(e),
          e)
        );
      }
      function ne() {
        return Z.startsWith("data:application/octet-stream;base64,");
      }
      function te() {
        var e = Z;
        try {
          if (e == Z && S) return new Uint8Array(S);
          if (l) return l(e);
          throw "both async and sync fetching of the wasm failed";
        } catch (e) {
          ee(e);
        }
      }
      (Z = "ort-wasm-simd-threaded.wasm"), ne() || (Z = T(Z));
      var re = {
        883136: () => {
          i.jsepRunPromise = new Promise(function (e) {
            i.Eb = e;
          });
        },
        883231: (e) => {
          i.Eb(e);
        },
        883269: (e) => i.zb(e),
        883302: (e) => i.Bb(e),
        883334: (e, n, t) => {
          i.gb(e, n, t, !0);
        },
        883373: (e, n, t) => {
          i.gb(e, n, t);
        },
        883406: (e) => {
          i.Da("Abs", e, void 0);
        },
        883457: (e) => {
          i.Da("Neg", e, void 0);
        },
        883508: (e) => {
          i.Da("Floor", e, void 0);
        },
        883561: (e) => {
          i.Da("Ceil", e, void 0);
        },
        883613: (e) => {
          i.Da("Reciprocal", e, void 0);
        },
        883671: (e) => {
          i.Da("Sqrt", e, void 0);
        },
        883723: (e) => {
          i.Da("Exp", e, void 0);
        },
        883774: (e) => {
          i.Da("Erf", e, void 0);
        },
        883825: (e) => {
          i.Da("Sigmoid", e, void 0);
        },
        883880: (e) => {
          i.Da("Sin", e, void 0);
        },
        883931: (e) => {
          i.Da("Cos", e, void 0);
        },
        883982: (e) => {
          i.Da("Tan", e, void 0);
        },
        884033: (e) => {
          i.Da("Asin", e, void 0);
        },
        884085: (e) => {
          i.Da("Acos", e, void 0);
        },
        884137: (e) => {
          i.Da("Atan", e, void 0);
        },
        884189: (e) => {
          i.Da("Sinh", e, void 0);
        },
        884241: (e) => {
          i.Da("Cosh", e, void 0);
        },
        884293: (e) => {
          i.Da("Asinh", e, void 0);
        },
        884346: (e) => {
          i.Da("Acosh", e, void 0);
        },
        884399: (e) => {
          i.Da("Atanh", e, void 0);
        },
        884452: (e, n, t) => {
          i.Da("ClipV10", e, { min: n, max: t });
        },
        884524: (e) => {
          i.Da("Clip", e, void 0);
        },
        884576: (e, n) => {
          i.Da("Elu", e, { alpha: n });
        },
        884634: (e) => {
          i.Da("Relu", e, void 0);
        },
        884686: (e, n) => {
          i.Da("LeakyRelu", e, { alpha: n });
        },
        884750: (e, n) => {
          i.Da("ThresholdedRelu", e, { alpha: n });
        },
        884820: (e) => {
          i.Da("Add", e, void 0);
        },
        884871: (e) => {
          i.Da("Sub", e, void 0);
        },
        884922: (e) => {
          i.Da("Mul", e, void 0);
        },
        884973: (e) => {
          i.Da("Div", e, void 0);
        },
        885024: (e) => {
          i.Da("Pow", e, void 0);
        },
        885075: (e, n, t) => {
          i.Da("Transpose", e, {
            perm: n ? Array.from(r().subarray(t >>> 0, (t + n) >>> 0)) : [],
          });
        },
        885188: (e, t, r, a, o, u, s, f, c, l) => {
          i.Da("Conv", e, {
            format: c ? "NHWC" : "NCHW",
            auto_pad: t,
            dilations: [r],
            group: a,
            kernel_shape: [o],
            pads: [u, s],
            strides: [f],
            w_is_const: () => !!n()[l >>> 0],
          });
        },
        885416: (e, t, r, a, o, u, s, f, c, l, p, d, m, h, _) => {
          i.Da("Conv", e, {
            format: h ? "NHWC" : "NCHW",
            auto_pad: t,
            dilations: [r, a],
            group: o,
            kernel_shape: [u, s],
            pads: [f, c, l, p],
            strides: [d, m],
            w_is_const: () => !!n()[_ >>> 0],
          });
        },
        885675: (e, t, r, a, o, u, s, f, c, l) => {
          i.Da("Conv", e, {
            format: c ? "NHWC" : "NCHW",
            auto_pad: t,
            dilations: [r],
            group: a,
            kernel_shape: [o],
            pads: [u, s],
            strides: [f],
            w_is_const: () => !!n()[l >>> 0],
          });
        },
        885903: (e, t, r, a, o, u, s, f, c, l, p, d, m, h, _) => {
          i.Da("Conv", e, {
            format: h ? "NHWC" : "NCHW",
            auto_pad: t,
            dilations: [r, a],
            group: o,
            kernel_shape: [u, s],
            pads: [f, c, l, p],
            strides: [d, m],
            w_is_const: () => !!n()[_ >>> 0],
          });
        },
        886162: (e, n) => {
          i.Da("GlobalAveragePool", e, { format: n ? "NHWC" : "NCHW" });
        },
        886253: (e, n, t, r, a, o, u, s, f, c, l, p, d, m, h, _) => {
          i.Da("AveragePool", e, {
            format: _ ? "NHWC" : "NCHW",
            auto_pad: n,
            ceil_mode: t,
            count_include_pad: r,
            storage_order: a,
            dilations: [o, u],
            kernel_shape: [s, f],
            pads: [c, l, p, d],
            strides: [m, h],
          });
        },
        886537: (e, n) => {
          i.Da("GlobalAveragePool", e, { format: n ? "NHWC" : "NCHW" });
        },
        886628: (e, n, t, r, a, o, u, s, f, c, l, p, d, m, h, _) => {
          i.Da("AveragePool", e, {
            format: _ ? "NHWC" : "NCHW",
            auto_pad: n,
            ceil_mode: t,
            count_include_pad: r,
            storage_order: a,
            dilations: [o, u],
            kernel_shape: [s, f],
            pads: [c, l, p, d],
            strides: [m, h],
          });
        },
        886912: (e, n) => {
          i.Da("GlobalMaxPool", e, { format: n ? "NHWC" : "NCHW" });
        },
        886999: (e, n, t, r, a, o, u, s, f, c, l, p, d, m, h, _) => {
          i.Da("MaxPool", e, {
            format: _ ? "NHWC" : "NCHW",
            auto_pad: n,
            ceil_mode: t,
            count_include_pad: r,
            storage_order: a,
            dilations: [o, u],
            kernel_shape: [s, f],
            pads: [c, l, p, d],
            strides: [m, h],
          });
        },
        887279: (e, n) => {
          i.Da("GlobalMaxPool", e, { format: n ? "NHWC" : "NCHW" });
        },
        887366: (e, n, t, r, a, o, u, s, f, c, l, p, d, m, h, _) => {
          i.Da("MaxPool", e, {
            format: _ ? "NHWC" : "NCHW",
            auto_pad: n,
            ceil_mode: t,
            count_include_pad: r,
            storage_order: a,
            dilations: [o, u],
            kernel_shape: [s, f],
            pads: [c, l, p, d],
            strides: [m, h],
          });
        },
        887646: (e, n, t, r, a) => {
          i.Da("Gemm", e, { alpha: n, beta: t, transA: r, transB: a });
        },
        887750: (e) => {
          i.Da("MatMul", e, void 0);
        },
        887804: (e) => {
          i.Cb(e);
        },
        887838: (e, n) => i.Db(e, n),
      };
      function ae(e) {
        (this.name = "ExitStatus"),
          (this.message = "Program terminated with exit(" + e + ")"),
          (this.status = e);
      }
      function oe(e) {
        (e = ce.Ua[e]) || ee(), ce.ib(e);
      }
      function ie(e) {
        var n = ce.xb();
        if (!n) return 6;
        ce.$a.push(n), (ce.Ua[e.Ta] = n), (n.Ta = e.Ta);
        var t = {
          cmd: "run",
          start_routine: e.Hb,
          arg: e.tb,
          pthread_ptr: e.Ta,
        };
        return (
          (n.Za = () => {
            (t.time = performance.now()), n.postMessage(t, e.Nb);
          }),
          n.loaded && (n.Za(), delete n.Za),
          0
        );
      }
      function ue(e) {
        if (w) return Ie(1, 1, e);
        x || (ce.Ib(), i.onExit && i.onExit(e), (U = !0)), b(e, new ae(e));
      }
      function se(e, n) {
        if (!n && w) throw (pe(e), "unwind");
        ue(e);
      }
      function fe(e) {
        e instanceof ae || "unwind" == e || b(1, e);
      }
      var ce = {
        Xa: [],
        $a: [],
        lb: [],
        Ua: {},
        cb: function () {
          w && ce.yb();
        },
        Pb: function () {},
        yb: function () {
          (ce.receiveObjectTransfer = ce.Gb),
            (ce.threadInitTLS = ce.kb),
            (ce.setExitStatus = ce.jb),
            (x = !1);
        },
        jb: function () {},
        Ib: function () {
          for (var e of Object.values(ce.Ua)) ce.ib(e);
          for (e of ce.Xa) e.terminate();
          ce.Xa = [];
        },
        ib: function (e) {
          var n = e.Ta;
          delete ce.Ua[n],
            ce.Xa.push(e),
            ce.$a.splice(ce.$a.indexOf(e), 1),
            (e.Ta = 0),
            wn(n);
        },
        Gb: function () {},
        kb: function () {
          ce.lb.forEach((e) => e());
        },
        Fb: function (e, n) {
          (e.onmessage = (t) => {
            var r = (t = t.data).cmd;
            if (
              (e.Ta && (ce.vb = e.Ta), t.targetThread && t.targetThread != hn())
            ) {
              var a = ce.Ua[t.Rb];
              a
                ? a.postMessage(t, t.transferList)
                : M(
                    'Internal error! Worker sent a message "' +
                      r +
                      '" to target pthread ' +
                      t.targetThread +
                      ", but that thread no longer exists!"
                  );
            } else
              "processProxyingQueue" === r
                ? xe(t.queue)
                : "spawnThread" === r
                ? ie(t)
                : "cleanupThread" === r
                ? oe(t.thread)
                : "killThread" === r
                ? ((t = t.thread),
                  (r = ce.Ua[t]),
                  delete ce.Ua[t],
                  r.terminate(),
                  wn(t),
                  ce.$a.splice(ce.$a.indexOf(r), 1),
                  (r.Ta = 0))
                : "cancelThread" === r
                ? ce.Ua[t.thread].postMessage({ cmd: "cancel" })
                : "loaded" === r
                ? ((e.loaded = !0), n && n(e), e.Za && (e.Za(), delete e.Za))
                : "print" === r
                ? C("Thread " + t.threadId + ": " + t.text)
                : "printErr" === r
                ? M("Thread " + t.threadId + ": " + t.text)
                : "alert" === r
                ? alert("Thread " + t.threadId + ": " + t.text)
                : "setimmediate" === t.target
                ? e.postMessage(t)
                : "onAbort" === r
                ? i.onAbort && i.onAbort(t.arg)
                : r && M("worker sent an unknown command " + r);
            ce.vb = void 0;
          }),
            (e.onerror = (e) => {
              throw (
                (M(
                  "worker sent an error! " +
                    e.filename +
                    ":" +
                    e.lineno +
                    ": " +
                    e.message
                ),
                e)
              );
            }),
            v &&
              (e.on("message", function (n) {
                e.onmessage({ data: n });
              }),
              e.on("error", function (n) {
                e.onerror(n);
              }),
              e.on("detachedExit", function () {})),
            e.postMessage({
              cmd: "load",
              urlOrBlob: i.mainScriptUrlOrBlob || _scriptDir,
              wasmMemory: R,
              wasmModule: k,
            });
        },
        sb: function () {
          var e = T("ort-wasm-simd-threaded.worker.js");
          ce.Xa.push(new Worker(e));
        },
        xb: function () {
          return 0 == ce.Xa.length && (ce.sb(), ce.Fb(ce.Xa[0])), ce.Xa.pop();
        },
      };
      function le(e) {
        for (; 0 < e.length; ) e.shift()(i);
      }
      function pe(e) {
        if (w) return Ie(2, 0, e);
        try {
          se(e);
        } catch (e) {
          fe(e);
        }
      }
      function de(e) {
        (this.Ya = e - 24),
          (this.rb = function (e) {
            a()[((this.Ya + 4) >> 2) >>> 0] = e;
          }),
          (this.ob = function (e) {
            a()[((this.Ya + 8) >> 2) >>> 0] = e;
          }),
          (this.pb = function () {
            r()[(this.Ya >> 2) >>> 0] = 0;
          }),
          (this.nb = function () {
            n()[((this.Ya + 12) >> 0) >>> 0] = 0;
          }),
          (this.qb = function () {
            n()[((this.Ya + 13) >> 0) >>> 0] = 0;
          }),
          (this.cb = function (e, n) {
            this.mb(), this.rb(e), this.ob(n), this.pb(), this.nb(), this.qb();
          }),
          (this.mb = function () {
            a()[((this.Ya + 16) >> 2) >>> 0] = 0;
          });
      }
      function me(e, n, t, r) {
        return w ? Ie(3, 1, e, n, t, r) : he(e, n, t, r);
      }
      function he(e, n, t, r) {
        if ("undefined" == typeof SharedArrayBuffer)
          return (
            M(
              "Current environment does not support SharedArrayBuffer, pthreads are not available!"
            ),
            6
          );
        var a = [];
        return w && 0 === a.length
          ? me(e, n, t, r)
          : ((e = { Hb: t, Ta: e, tb: r, Nb: a }),
            w ? ((e.Ob = "spawnThread"), postMessage(e, a), 0) : ie(e));
      }
      function _e(e, n, t) {
        return w ? Ie(4, 1, e, n, t) : 0;
      }
      function be(e, n) {
        if (w) return Ie(5, 1, e, n);
      }
      function ge(e, n) {
        if (w) return Ie(6, 1, e, n);
      }
      function ye(e, n, t) {
        if (w) return Ie(7, 1, e, n, t);
      }
      function ve(e, n, t) {
        return w ? Ie(8, 1, e, n, t) : 0;
      }
      function we(e, n) {
        if (w) return Ie(9, 1, e, n);
      }
      function De(e, n, t) {
        if (w) return Ie(10, 1, e, n, t);
      }
      function Te(e, n, t, r) {
        if (w) return Ie(11, 1, e, n, t, r);
      }
      function Oe(e, n, t, r) {
        if (w) return Ie(12, 1, e, n, t, r);
      }
      function Ae(e, n, t, r) {
        if (w) return Ie(13, 1, e, n, t, r);
      }
      function Se(e) {
        if (w) return Ie(14, 1, e);
      }
      function Ce(e, n) {
        if (w) return Ie(15, 1, e, n);
      }
      function Me(e, n, t) {
        if (w) return Ie(16, 1, e, n, t);
      }
      function xe(e) {
        Atomics.store(r(), e >> 2, 1),
          hn() && vn(e),
          Atomics.compareExchange(r(), e >> 2, 1, 0);
      }
      function Re(e) {
        return a()[e >>> 2] + 4294967296 * r()[(e + 4) >>> 2];
      }
      function ke(e, n, t, r, a, o) {
        return w ? Ie(17, 1, e, n, t, r, a, o) : -52;
      }
      function Ee(e, n, t, r, a, o) {
        if (w) return Ie(18, 1, e, n, t, r, a, o);
      }
      function We(e) {
        var t = G(e) + 1,
          r = _n(t);
        return r && B(e, n(), r, t), r;
      }
      function He(e, n, t) {
        function o(e) {
          return (e = e.toTimeString().match(/\(([A-Za-z ]+)\)$/))
            ? e[1]
            : "GMT";
        }
        if (w) return Ie(19, 1, e, n, t);
        var i = new Date().getFullYear(),
          u = new Date(i, 0, 1),
          s = new Date(i, 6, 1);
        i = u.getTimezoneOffset();
        var f = s.getTimezoneOffset(),
          c = Math.max(i, f);
        (r()[(e >> 2) >>> 0] = 60 * c),
          (r()[(n >> 2) >>> 0] = Number(i != f)),
          (e = o(u)),
          (n = o(s)),
          (e = We(e)),
          (n = We(n)),
          f < i
            ? ((a()[(t >> 2) >>> 0] = e), (a()[((t + 4) >> 2) >>> 0] = n))
            : ((a()[(t >> 2) >>> 0] = n), (a()[((t + 4) >> 2) >>> 0] = e));
      }
      (i.PThread = ce),
        (i.establishStackSpace = function () {
          var e = hn(),
            n = r()[((e + 44) >> 2) >>> 0];
          (e = r()[((e + 48) >> 2) >>> 0]), Tn(n, n - e), An(n);
        }),
        (i.invokeEntryPoint = function (e, n) {
          (e = Mn.apply(null, [e, n])), x ? ce.jb(e) : Dn(e);
        }),
        (i.executeNotifiedProxyingQueue = xe);
      var Fe,
        Ne,
        Pe = [];
      function Ue(e, n, a) {
        var i;
        for (Pe.length = 0, a >>= 2; (i = t()[n++ >>> 0]); )
          (a += (105 != i) & a),
            Pe.push(105 == i ? r()[a >>> 0] : o()[a++ >>> 1]),
            ++a;
        return re[e].apply(null, Pe);
      }
      function Ie(e, n) {
        var t = arguments.length - 2,
          r = arguments;
        return (function (e) {
          var n = On();
          return (e = e()), An(n), e;
        })(() => {
          for (var a = Sn(8 * t), i = a >> 3, u = 0; u < t; u++) {
            var s = r[2 + u];
            o()[(i + u) >>> 0] = s;
          }
          return yn(e, t, a, n);
        });
      }
      Ne = v
        ? () => {
            var e = process.hrtime();
            return 1e3 * e[0] + e[1] / 1e6;
          }
        : w
        ? () => performance.now() - i.__performance_now_clock_drift
        : () => performance.now();
      var Ye,
        je = [],
        Be = {};
      function Ge() {
        if (!Ye) {
          var e,
            n = {
              USER: "web_user",
              LOGNAME: "web_user",
              PATH: "/",
              PWD: "/",
              HOME: "/home/web_user",
              LANG:
                (
                  ("object" == typeof navigator &&
                    navigator.languages &&
                    navigator.languages[0]) ||
                  "C"
                ).replace("-", "_") + ".UTF-8",
              _: _ || "./this.program",
            };
          for (e in Be) void 0 === Be[e] ? delete n[e] : (n[e] = Be[e]);
          var t = [];
          for (e in n) t.push(e + "=" + n[e]);
          Ye = t;
        }
        return Ye;
      }
      function qe(e, t) {
        if (w) return Ie(20, 1, e, t);
        var r = 0;
        return (
          Ge().forEach(function (o, i) {
            var u = t + r;
            for (
              i = a()[((e + 4 * i) >> 2) >>> 0] = u, u = 0;
              u < o.length;
              ++u
            )
              n()[(i++ >> 0) >>> 0] = o.charCodeAt(u);
            (n()[(i >> 0) >>> 0] = 0), (r += o.length + 1);
          }),
          0
        );
      }
      function ze(e, n) {
        if (w) return Ie(21, 1, e, n);
        var t = Ge();
        a()[(e >> 2) >>> 0] = t.length;
        var r = 0;
        return (
          t.forEach(function (e) {
            r += e.length + 1;
          }),
          (a()[(n >> 2) >>> 0] = r),
          0
        );
      }
      function Le(e) {
        return w ? Ie(22, 1, e) : 52;
      }
      function Ve(e, n, t, r) {
        return w ? Ie(23, 1, e, n, t, r) : 52;
      }
      function Xe(e, n, t, r, a) {
        return w ? Ie(24, 1, e, n, t, r, a) : 70;
      }
      var Je = [null, [], []];
      function Ze(e, n, r, o) {
        if (w) return Ie(25, 1, e, n, r, o);
        for (var i = 0, u = 0; u < r; u++) {
          var s = a()[(n >> 2) >>> 0],
            f = a()[((n + 4) >> 2) >>> 0];
          n += 8;
          for (var c = 0; c < f; c++) {
            var l = t()[(s + c) >>> 0],
              p = Je[e];
            0 === l || 10 === l
              ? ((1 === e ? C : M)(Y(p, 0)), (p.length = 0))
              : p.push(l);
          }
          i += f;
        }
        return (a()[(o >> 2) >>> 0] = i), 0;
      }
      function $e(e) {
        return 0 == e % 4 && (0 != e % 100 || 0 == e % 400);
      }
      var Qe = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        Ke = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
      function en(e, t, a, o) {
        function i(e, n, t) {
          for (
            e = "number" == typeof e ? e.toString() : e || "";
            e.length < n;

          )
            e = t[0] + e;
          return e;
        }
        function u(e, n) {
          return i(e, n, "0");
        }
        function s(e, n) {
          function t(e) {
            return 0 > e ? -1 : 0 < e ? 1 : 0;
          }
          var r;
          return (
            0 === (r = t(e.getFullYear() - n.getFullYear())) &&
              0 === (r = t(e.getMonth() - n.getMonth())) &&
              (r = t(e.getDate() - n.getDate())),
            r
          );
        }
        function f(e) {
          switch (e.getDay()) {
            case 0:
              return new Date(e.getFullYear() - 1, 11, 29);
            case 1:
              return e;
            case 2:
              return new Date(e.getFullYear(), 0, 3);
            case 3:
              return new Date(e.getFullYear(), 0, 2);
            case 4:
              return new Date(e.getFullYear(), 0, 1);
            case 5:
              return new Date(e.getFullYear() - 1, 11, 31);
            case 6:
              return new Date(e.getFullYear() - 1, 11, 30);
          }
        }
        function c(e) {
          var n = e.Va;
          for (e = new Date(new Date(e.Wa + 1900, 0, 1).getTime()); 0 < n; ) {
            var t = e.getMonth(),
              r = ($e(e.getFullYear()) ? Qe : Ke)[t];
            if (!(n > r - e.getDate())) {
              e.setDate(e.getDate() + n);
              break;
            }
            (n -= r - e.getDate() + 1),
              e.setDate(1),
              11 > t
                ? e.setMonth(t + 1)
                : (e.setMonth(0), e.setFullYear(e.getFullYear() + 1));
          }
          return (
            (t = new Date(e.getFullYear() + 1, 0, 4)),
            (n = f(new Date(e.getFullYear(), 0, 4))),
            (t = f(t)),
            0 >= s(n, e)
              ? 0 >= s(t, e)
                ? e.getFullYear() + 1
                : e.getFullYear()
              : e.getFullYear() - 1
          );
        }
        var l = r()[((o + 40) >> 2) >>> 0];
        for (var p in ((o = {
          Lb: r()[(o >> 2) >>> 0],
          Kb: r()[((o + 4) >> 2) >>> 0],
          ab: r()[((o + 8) >> 2) >>> 0],
          fb: r()[((o + 12) >> 2) >>> 0],
          bb: r()[((o + 16) >> 2) >>> 0],
          Wa: r()[((o + 20) >> 2) >>> 0],
          Sa: r()[((o + 24) >> 2) >>> 0],
          Va: r()[((o + 28) >> 2) >>> 0],
          Sb: r()[((o + 32) >> 2) >>> 0],
          Jb: r()[((o + 36) >> 2) >>> 0],
          Mb: l ? j(l) : "",
        }),
        (a = j(a)),
        (l = {
          "%c": "%a %b %d %H:%M:%S %Y",
          "%D": "%m/%d/%y",
          "%F": "%Y-%m-%d",
          "%h": "%b",
          "%r": "%I:%M:%S %p",
          "%R": "%H:%M",
          "%T": "%H:%M:%S",
          "%x": "%m/%d/%y",
          "%X": "%H:%M:%S",
          "%Ec": "%c",
          "%EC": "%C",
          "%Ex": "%m/%d/%y",
          "%EX": "%H:%M:%S",
          "%Ey": "%y",
          "%EY": "%Y",
          "%Od": "%d",
          "%Oe": "%e",
          "%OH": "%H",
          "%OI": "%I",
          "%Om": "%m",
          "%OM": "%M",
          "%OS": "%S",
          "%Ou": "%u",
          "%OU": "%U",
          "%OV": "%V",
          "%Ow": "%w",
          "%OW": "%W",
          "%Oy": "%y",
        })))
          a = a.replace(new RegExp(p, "g"), l[p]);
        var d =
            "Sunday Monday Tuesday Wednesday Thursday Friday Saturday".split(
              " "
            ),
          m =
            "January February March April May June July August September October November December".split(
              " "
            );
        for (p in ((l = {
          "%a": function (e) {
            return d[e.Sa].substring(0, 3);
          },
          "%A": function (e) {
            return d[e.Sa];
          },
          "%b": function (e) {
            return m[e.bb].substring(0, 3);
          },
          "%B": function (e) {
            return m[e.bb];
          },
          "%C": function (e) {
            return u(((e.Wa + 1900) / 100) | 0, 2);
          },
          "%d": function (e) {
            return u(e.fb, 2);
          },
          "%e": function (e) {
            return i(e.fb, 2, " ");
          },
          "%g": function (e) {
            return c(e).toString().substring(2);
          },
          "%G": function (e) {
            return c(e);
          },
          "%H": function (e) {
            return u(e.ab, 2);
          },
          "%I": function (e) {
            return 0 == (e = e.ab) ? (e = 12) : 12 < e && (e -= 12), u(e, 2);
          },
          "%j": function (e) {
            for (
              var n = 0, t = 0;
              t <= e.bb - 1;
              n += ($e(e.Wa + 1900) ? Qe : Ke)[t++]
            );
            return u(e.fb + n, 3);
          },
          "%m": function (e) {
            return u(e.bb + 1, 2);
          },
          "%M": function (e) {
            return u(e.Kb, 2);
          },
          "%n": function () {
            return "\n";
          },
          "%p": function (e) {
            return 0 <= e.ab && 12 > e.ab ? "AM" : "PM";
          },
          "%S": function (e) {
            return u(e.Lb, 2);
          },
          "%t": function () {
            return "\t";
          },
          "%u": function (e) {
            return e.Sa || 7;
          },
          "%U": function (e) {
            return u(Math.floor((e.Va + 7 - e.Sa) / 7), 2);
          },
          "%V": function (e) {
            var n = Math.floor((e.Va + 7 - ((e.Sa + 6) % 7)) / 7);
            if ((2 >= (e.Sa + 371 - e.Va - 2) % 7 && n++, n))
              53 == n &&
                (4 == (t = (e.Sa + 371 - e.Va) % 7) ||
                  (3 == t && $e(e.Wa)) ||
                  (n = 1));
            else {
              n = 52;
              var t = (e.Sa + 7 - e.Va - 1) % 7;
              (4 == t || (5 == t && $e((e.Wa % 400) - 1))) && n++;
            }
            return u(n, 2);
          },
          "%w": function (e) {
            return e.Sa;
          },
          "%W": function (e) {
            return u(Math.floor((e.Va + 7 - ((e.Sa + 6) % 7)) / 7), 2);
          },
          "%y": function (e) {
            return (e.Wa + 1900).toString().substring(2);
          },
          "%Y": function (e) {
            return e.Wa + 1900;
          },
          "%z": function (e) {
            var n = 0 <= (e = e.Jb);
            return (
              (e = Math.abs(e) / 60),
              (n ? "+" : "-") +
                String("0000" + ((e / 60) * 100 + (e % 60))).slice(-4)
            );
          },
          "%Z": function (e) {
            return e.Mb;
          },
          "%%": function () {
            return "%";
          },
        }),
        (a = a.replace(/%%/g, "\0\0")),
        l))
          a.includes(p) && (a = a.replace(new RegExp(p, "g"), l[p](o)));
        return (
          (p = (function (e) {
            var n = Array(G(e) + 1);
            return B(e, n, 0, n.length), n;
          })((a = a.replace(/\0\0/g, "%")))),
          p.length > t
            ? 0
            : ((function (e, t) {
                n().set(e, t >>> 0);
              })(p, e),
              p.length - 1)
        );
      }
      function nn(e) {
        try {
          e();
        } catch (e) {
          ee(e);
        }
      }
      var tn = 0,
        rn = null,
        an = 0,
        on = [],
        un = {},
        sn = {},
        fn = 0,
        cn = null,
        ln = [];
      function pn(e) {
        var n,
          t = {};
        for (n in e)
          !(function (n) {
            var r = e[n];
            t[n] =
              "function" == typeof r
                ? function () {
                    on.push(n);
                    try {
                      return r.apply(null, arguments);
                    } finally {
                      U ||
                        (on.pop() === n || ee(),
                        rn &&
                          1 === tn &&
                          0 === on.length &&
                          ((tn = 0),
                          nn(Rn),
                          "undefined" != typeof Fibers && Fibers.Tb()));
                    }
                  }
                : r;
          })(n);
        return t;
      }
      ce.cb();
      var dn = [
          null,
          ue,
          pe,
          me,
          _e,
          be,
          ge,
          ye,
          ve,
          we,
          De,
          Te,
          Oe,
          Ae,
          Se,
          Ce,
          Me,
          ke,
          Ee,
          He,
          qe,
          ze,
          Le,
          Ve,
          Xe,
          Ze,
        ],
        mn = {
          t: function (e, n, t) {
            return (function (e) {
              return (function (e) {
                if (!U) {
                  if (0 === tn) {
                    var n = !1,
                      t = !1;
                    e((e) => {
                      if (!U && ((an = e || 0), (n = !0), t)) {
                        (tn = 2),
                          nn(() => kn(rn)),
                          "undefined" != typeof Browser &&
                            Browser.eb.wb &&
                            Browser.eb.resume(),
                          (e = !1);
                        try {
                          var a = (function () {
                            var e = r()[((rn + 8) >> 2) >>> 0];
                            return (0, i.asm[sn[e]])();
                          })();
                        } catch (n) {
                          (a = n), (e = !0);
                        }
                        var o = !1;
                        if (!rn) {
                          var u = cn;
                          u &&
                            ((cn = null),
                            (e ? u.reject : u.resolve)(a),
                            (o = !0));
                        }
                        if (e && !o) throw a;
                      }
                    }),
                      (t = !0),
                      n ||
                        ((tn = 1),
                        (rn = (function () {
                          var e = _n(65548),
                            n = e + 12;
                          (r()[(e >> 2) >>> 0] = n),
                            (r()[((e + 4) >> 2) >>> 0] = n + 65536),
                            (n = on[0]);
                          var t = un[n];
                          return (
                            void 0 === t &&
                              ((t = fn++), (un[n] = t), (sn[t] = n)),
                            (n = t),
                            (r()[((e + 8) >> 2) >>> 0] = n),
                            e
                          );
                        })()),
                        "undefined" != typeof Browser &&
                          Browser.eb.wb &&
                          Browser.eb.pause(),
                        nn(() => xn(rn)));
                  } else
                    2 === tn
                      ? ((tn = 0),
                        nn(En),
                        bn(rn),
                        (rn = null),
                        ln.forEach((e) => {
                          if (!U)
                            try {
                              e();
                            } catch (e) {
                              fe(e);
                            }
                        }))
                      : ee("invalid state: " + tn);
                  return an;
                }
              })((n) => {
                e().then(n);
              });
            })(async () => {
              await i.Ab(e, n, t);
            });
          },
          b: function (e) {
            return _n(e + 24) + 24;
          },
          c: function (e, n, t) {
            throw (new de(e).cb(n, t), e);
          },
          M: function (e) {
            gn(e, !y, 1, !g), ce.kb();
          },
          m: function (e) {
            w ? postMessage({ cmd: "cleanupThread", thread: e }) : oe(e);
          },
          E: he,
          j: _e,
          S: be,
          B: ge,
          D: ye,
          U: ve,
          Q: we,
          J: De,
          P: Te,
          q: Oe,
          C: Ae,
          z: Se,
          R: Ce,
          A: Me,
          $: function () {},
          s: function () {
            ee(
              "To use dlopen, you need enable dynamic linking, see https://github.com/emscripten-core/emscripten/wiki/Linking"
            );
          },
          aa: function () {
            ee(
              "To use dlopen, you need enable dynamic linking, see https://github.com/emscripten-core/emscripten/wiki/Linking"
            );
          },
          r: function () {
            return Date.now();
          },
          F: function () {
            return 2097152;
          },
          W: function () {
            return !0;
          },
          G: function (e, n, t, r) {
            if (e == n) setTimeout(() => xe(r));
            else if (w)
              postMessage({
                targetThread: e,
                cmd: "processProxyingQueue",
                queue: r,
              });
            else {
              if (!(e = ce.Ua[e])) return;
              e.postMessage({ cmd: "processProxyingQueue", queue: r });
            }
            return 1;
          },
          L: function () {
            return -1;
          },
          X: function (e, n) {
            (e = new Date(1e3 * Re(e))),
              (r()[(n >> 2) >>> 0] = e.getUTCSeconds()),
              (r()[((n + 4) >> 2) >>> 0] = e.getUTCMinutes()),
              (r()[((n + 8) >> 2) >>> 0] = e.getUTCHours()),
              (r()[((n + 12) >> 2) >>> 0] = e.getUTCDate()),
              (r()[((n + 16) >> 2) >>> 0] = e.getUTCMonth()),
              (r()[((n + 20) >> 2) >>> 0] = e.getUTCFullYear() - 1900),
              (r()[((n + 24) >> 2) >>> 0] = e.getUTCDay()),
              (e =
                ((e.getTime() -
                  Date.UTC(e.getUTCFullYear(), 0, 1, 0, 0, 0, 0)) /
                  864e5) |
                0),
              (r()[((n + 28) >> 2) >>> 0] = e);
          },
          Y: function (e, n) {
            (e = new Date(1e3 * Re(e))),
              (r()[(n >> 2) >>> 0] = e.getSeconds()),
              (r()[((n + 4) >> 2) >>> 0] = e.getMinutes()),
              (r()[((n + 8) >> 2) >>> 0] = e.getHours()),
              (r()[((n + 12) >> 2) >>> 0] = e.getDate()),
              (r()[((n + 16) >> 2) >>> 0] = e.getMonth()),
              (r()[((n + 20) >> 2) >>> 0] = e.getFullYear() - 1900),
              (r()[((n + 24) >> 2) >>> 0] = e.getDay());
            var t = new Date(e.getFullYear(), 0, 1),
              a = ((e.getTime() - t.getTime()) / 864e5) | 0;
            (r()[((n + 28) >> 2) >>> 0] = a),
              (r()[((n + 36) >> 2) >>> 0] = -60 * e.getTimezoneOffset()),
              (a = new Date(e.getFullYear(), 6, 1).getTimezoneOffset()),
              (e =
                0 |
                (a != (t = t.getTimezoneOffset()) &&
                  e.getTimezoneOffset() == Math.min(t, a))),
              (r()[((n + 32) >> 2) >>> 0] = e);
          },
          Z: function (e) {
            var n = new Date(
                r()[((e + 20) >> 2) >>> 0] + 1900,
                r()[((e + 16) >> 2) >>> 0],
                r()[((e + 12) >> 2) >>> 0],
                r()[((e + 8) >> 2) >>> 0],
                r()[((e + 4) >> 2) >>> 0],
                r()[(e >> 2) >>> 0],
                0
              ),
              t = r()[((e + 32) >> 2) >>> 0],
              a = n.getTimezoneOffset(),
              o = new Date(n.getFullYear(), 0, 1),
              i = new Date(n.getFullYear(), 6, 1).getTimezoneOffset(),
              u = o.getTimezoneOffset(),
              s = Math.min(u, i);
            return (
              0 > t
                ? (r()[((e + 32) >> 2) >>> 0] = Number(i != u && s == a))
                : 0 < t != (s == a) &&
                  ((i = Math.max(u, i)),
                  n.setTime(n.getTime() + 6e4 * ((0 < t ? s : i) - a))),
              (r()[((e + 24) >> 2) >>> 0] = n.getDay()),
              (t = ((n.getTime() - o.getTime()) / 864e5) | 0),
              (r()[((e + 28) >> 2) >>> 0] = t),
              (r()[(e >> 2) >>> 0] = n.getSeconds()),
              (r()[((e + 4) >> 2) >>> 0] = n.getMinutes()),
              (r()[((e + 8) >> 2) >>> 0] = n.getHours()),
              (r()[((e + 12) >> 2) >>> 0] = n.getDate()),
              (r()[((e + 16) >> 2) >>> 0] = n.getMonth()),
              (n.getTime() / 1e3) | 0
            );
          },
          H: ke,
          I: Ee,
          _: function e(n, t, r) {
            e.ub || ((e.ub = !0), He(n, t, r));
          },
          e: function () {
            ee("");
          },
          d: Ue,
          l: Ue,
          n: function () {
            if (!v && !y) {
              var e =
                "Blocking on the main thread is very dangerous, see https://emscripten.org/docs/porting/pthreads.html#blocking-on-the-main-browser-thread";
              Fe || (Fe = {}),
                Fe[e] || ((Fe[e] = 1), v && (e = "warning: " + e), M(e));
            }
          },
          y: function () {
            return 4294901760;
          },
          g: Ne,
          T: function (e, n, r) {
            t().copyWithin(e >>> 0, n >>> 0, (n + r) >>> 0);
          },
          h: function () {
            return v
              ? require("os").cpus().length
              : navigator.hardwareConcurrency;
          },
          K: function (e, n, t) {
            (je.length = n), (t >>= 3);
            for (var r = 0; r < n; r++) je[r] = o()[(t + r) >>> 0];
            return (0 > e ? re[-e - 1] : dn[e]).apply(null, je);
          },
          w: function (e) {
            var n = t().length;
            if ((e >>>= 0) <= n || 4294901760 < e) return !1;
            for (var r = 1; 4 >= r; r *= 2) {
              var a = n * (1 + 0.2 / r);
              a = Math.min(a, e + 100663296);
              var o = Math;
              (a = Math.max(e, a)),
                (o = o.min.call(
                  o,
                  4294901760,
                  a + ((65536 - (a % 65536)) % 65536)
                ));
              e: {
                try {
                  R.grow((o - E.byteLength + 65535) >>> 16), q(R.buffer);
                  var i = 1;
                  break e;
                } catch (e) {}
                i = void 0;
              }
              if (i) return !0;
            }
            return !1;
          },
          V: function () {
            throw "unwind";
          },
          N: qe,
          O: ze,
          k: se,
          i: Le,
          p: Ve,
          u: Xe,
          o: Ze,
          v: function e(t, r) {
            e.hb ||
              (e.hb = (function () {
                if (
                  "object" == typeof crypto &&
                  "function" == typeof crypto.getRandomValues
                ) {
                  var e = new Uint8Array(1);
                  return () => (crypto.getRandomValues(e), e[0]);
                }
                if (v)
                  try {
                    var n = require("crypto");
                    return () => n.randomBytes(1)[0];
                  } catch (e) {}
                return () => ee("randomDevice");
              })());
            for (var a = 0; a < r; a++) n()[((t + a) >> 0) >>> 0] = e.hb();
            return 0;
          },
          a: R || i.wasmMemory,
          x: en,
          f: function (e, n, t, r) {
            return en(e, n, t, r);
          },
        };
      !(function () {
        function e(e, n) {
          (e = pn((e = e.exports))),
            (i.asm = e),
            ce.lb.push(i.asm.Aa),
            V.unshift(i.asm.ba),
            (k = n),
            w ||
              ($--,
              i.monitorRunDependencies && i.monitorRunDependencies($),
              0 == $ &&
                (null !== Q && (clearInterval(Q), (Q = null)),
                K && ((n = K), (K = null), n())));
        }
        function n(n) {
          e(n.instance, n.module);
        }
        function t(e) {
          return (function () {
            if (!S && (g || y)) {
              if ("function" == typeof fetch && !Z.startsWith("file://"))
                return fetch(Z, { credentials: "same-origin" })
                  .then(function (e) {
                    if (!e.ok)
                      throw "failed to load wasm binary file at '" + Z + "'";
                    return e.arrayBuffer();
                  })
                  .catch(function () {
                    return te();
                  });
              if (c)
                return new Promise(function (e, n) {
                  c(
                    Z,
                    function (n) {
                      e(new Uint8Array(n));
                    },
                    n
                  );
                });
            }
            return Promise.resolve().then(function () {
              return te();
            });
          })()
            .then(function (e) {
              return WebAssembly.instantiate(e, r);
            })
            .then(function (e) {
              return e;
            })
            .then(e, function (e) {
              M("failed to asynchronously prepare wasm: " + e), ee(e);
            });
        }
        var r = { a: mn };
        if (
          (w || ($++, i.monitorRunDependencies && i.monitorRunDependencies($)),
          i.instantiateWasm)
        )
          try {
            return pn(i.instantiateWasm(r, e));
          } catch (e) {
            return (
              M("Module.instantiateWasm callback failed with error: " + e), !1
            );
          }
        (S ||
        "function" != typeof WebAssembly.instantiateStreaming ||
        ne() ||
        Z.startsWith("file://") ||
        v ||
        "function" != typeof fetch
          ? t(n)
          : fetch(Z, { credentials: "same-origin" }).then(function (e) {
              return WebAssembly.instantiateStreaming(e, r).then(
                n,
                function (e) {
                  return (
                    M("wasm streaming compile failed: " + e),
                    M("falling back to ArrayBuffer instantiation"),
                    t(n)
                  );
                }
              );
            })
        ).catch(s);
      })(),
        (i.___wasm_call_ctors = function () {
          return (i.___wasm_call_ctors = i.asm.ba).apply(null, arguments);
        }),
        (i._OrtInit = function () {
          return (i._OrtInit = i.asm.ca).apply(null, arguments);
        }),
        (i._OrtCreateSessionOptions = function () {
          return (i._OrtCreateSessionOptions = i.asm.da).apply(null, arguments);
        }),
        (i._OrtAppendExecutionProvider = function () {
          return (i._OrtAppendExecutionProvider = i.asm.ea).apply(
            null,
            arguments
          );
        }),
        (i._OrtAddSessionConfigEntry = function () {
          return (i._OrtAddSessionConfigEntry = i.asm.fa).apply(
            null,
            arguments
          );
        }),
        (i._OrtReleaseSessionOptions = function () {
          return (i._OrtReleaseSessionOptions = i.asm.ga).apply(
            null,
            arguments
          );
        }),
        (i._OrtCreateSession = function () {
          return (i._OrtCreateSession = i.asm.ha).apply(null, arguments);
        }),
        (i._OrtReleaseSession = function () {
          return (i._OrtReleaseSession = i.asm.ia).apply(null, arguments);
        }),
        (i._OrtGetInputCount = function () {
          return (i._OrtGetInputCount = i.asm.ja).apply(null, arguments);
        }),
        (i._OrtGetOutputCount = function () {
          return (i._OrtGetOutputCount = i.asm.ka).apply(null, arguments);
        }),
        (i._OrtGetInputName = function () {
          return (i._OrtGetInputName = i.asm.la).apply(null, arguments);
        }),
        (i._OrtGetOutputName = function () {
          return (i._OrtGetOutputName = i.asm.ma).apply(null, arguments);
        }),
        (i._OrtFree = function () {
          return (i._OrtFree = i.asm.na).apply(null, arguments);
        }),
        (i._OrtCreateTensor = function () {
          return (i._OrtCreateTensor = i.asm.oa).apply(null, arguments);
        }),
        (i._OrtGetTensorData = function () {
          return (i._OrtGetTensorData = i.asm.pa).apply(null, arguments);
        }),
        (i._OrtReleaseTensor = function () {
          return (i._OrtReleaseTensor = i.asm.qa).apply(null, arguments);
        }),
        (i._OrtCreateRunOptions = function () {
          return (i._OrtCreateRunOptions = i.asm.ra).apply(null, arguments);
        }),
        (i._OrtAddRunConfigEntry = function () {
          return (i._OrtAddRunConfigEntry = i.asm.sa).apply(null, arguments);
        }),
        (i._OrtReleaseRunOptions = function () {
          return (i._OrtReleaseRunOptions = i.asm.ta).apply(null, arguments);
        }),
        (i._OrtRun = function () {
          return (i._OrtRun = i.asm.ua).apply(null, arguments);
        }),
        (i._OrtEndProfiling = function () {
          return (i._OrtEndProfiling = i.asm.va).apply(null, arguments);
        }),
        (i._JsepOutput = function () {
          return (i._JsepOutput = i.asm.wa).apply(null, arguments);
        });
      var hn = (i._pthread_self = function () {
          return (hn = i._pthread_self = i.asm.xa).apply(null, arguments);
        }),
        _n = (i._malloc = function () {
          return (_n = i._malloc = i.asm.ya).apply(null, arguments);
        }),
        bn = (i._free = function () {
          return (bn = i._free = i.asm.za).apply(null, arguments);
        });
      i.__emscripten_tls_init = function () {
        return (i.__emscripten_tls_init = i.asm.Aa).apply(null, arguments);
      };
      var gn = (i.__emscripten_thread_init = function () {
        return (gn = i.__emscripten_thread_init = i.asm.Ba).apply(
          null,
          arguments
        );
      });
      i.__emscripten_thread_crashed = function () {
        return (i.__emscripten_thread_crashed = i.asm.Ca).apply(
          null,
          arguments
        );
      };
      var yn = (i._emscripten_run_in_main_runtime_thread_js = function () {
          return (yn = i._emscripten_run_in_main_runtime_thread_js =
            i.asm.Ea).apply(null, arguments);
        }),
        vn = (i.__emscripten_proxy_execute_task_queue = function () {
          return (vn = i.__emscripten_proxy_execute_task_queue =
            i.asm.Fa).apply(null, arguments);
        }),
        wn = (i.__emscripten_thread_free_data = function () {
          return (wn = i.__emscripten_thread_free_data = i.asm.Ga).apply(
            null,
            arguments
          );
        }),
        Dn = (i.__emscripten_thread_exit = function () {
          return (Dn = i.__emscripten_thread_exit = i.asm.Ha).apply(
            null,
            arguments
          );
        }),
        Tn = (i._emscripten_stack_set_limits = function () {
          return (Tn = i._emscripten_stack_set_limits = i.asm.Ia).apply(
            null,
            arguments
          );
        }),
        On = (i.stackSave = function () {
          return (On = i.stackSave = i.asm.Ja).apply(null, arguments);
        }),
        An = (i.stackRestore = function () {
          return (An = i.stackRestore = i.asm.Ka).apply(null, arguments);
        }),
        Sn = (i.stackAlloc = function () {
          return (Sn = i.stackAlloc = i.asm.La).apply(null, arguments);
        });
      i.___cxa_is_pointer_type = function () {
        return (i.___cxa_is_pointer_type = i.asm.Ma).apply(null, arguments);
      };
      var Cn,
        Mn = (i.dynCall_ii = function () {
          return (Mn = i.dynCall_ii = i.asm.Na).apply(null, arguments);
        }),
        xn = (i._asyncify_start_unwind = function () {
          return (xn = i._asyncify_start_unwind = i.asm.Oa).apply(
            null,
            arguments
          );
        }),
        Rn = (i._asyncify_stop_unwind = function () {
          return (Rn = i._asyncify_stop_unwind = i.asm.Pa).apply(
            null,
            arguments
          );
        }),
        kn = (i._asyncify_start_rewind = function () {
          return (kn = i._asyncify_start_rewind = i.asm.Qa).apply(
            null,
            arguments
          );
        }),
        En = (i._asyncify_stop_rewind = function () {
          return (En = i._asyncify_stop_rewind = i.asm.Ra).apply(
            null,
            arguments
          );
        });
      function Wn() {
        function e() {
          if (
            !Cn &&
            ((Cn = !0), (i.calledRun = !0), !U) &&
            (w || le(V),
            u(i),
            i.onRuntimeInitialized && i.onRuntimeInitialized(),
            !w)
          ) {
            if (i.postRun)
              for (
                "function" == typeof i.postRun && (i.postRun = [i.postRun]);
                i.postRun.length;

              ) {
                var e = i.postRun.shift();
                X.unshift(e);
              }
            le(X);
          }
        }
        if (!(0 < $))
          if (w) u(i), w || le(V), postMessage({ cmd: "loaded" });
          else {
            if (i.preRun)
              for (
                "function" == typeof i.preRun && (i.preRun = [i.preRun]);
                i.preRun.length;

              )
                J();
            le(L),
              0 < $ ||
                (i.setStatus
                  ? (i.setStatus("Running..."),
                    setTimeout(function () {
                      setTimeout(function () {
                        i.setStatus("");
                      }, 1),
                        e();
                    }, 1))
                  : e());
          }
      }
      if (
        ((i.___start_em_js = 887873),
        (i.___stop_em_js = 888034),
        (i.UTF8ToString = j),
        (i.stringToUTF8 = function (e, n, r) {
          return B(e, t(), n, r);
        }),
        (i.lengthBytesUTF8 = G),
        (i.keepRuntimeAlive = function () {
          return x;
        }),
        (i.wasmMemory = R),
        (i.stackSave = On),
        (i.stackRestore = An),
        (i.stackAlloc = Sn),
        (i.ExitStatus = ae),
        (i.PThread = ce),
        (K = function e() {
          Cn || Wn(), Cn || (K = e);
        }),
        i.preInit)
      )
        for (
          "function" == typeof i.preInit && (i.preInit = [i.preInit]);
          0 < i.preInit.length;

        )
          i.preInit.pop()();
      return Wn(), e.ready;
    });
"object" == typeof exports && "object" == typeof module
  ? (module.exports = e)
  : "function" == typeof define && define.amd
  ? define([], function () {
      return e;
    })
  : "object" == typeof exports && (exports.ortWasmThreaded = e);
