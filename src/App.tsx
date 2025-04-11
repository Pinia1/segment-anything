import { useEffect, useRef, useState } from "react";

import "./App.css";
import * as ort from "onnxruntime-web/webgpu";
const wasmBaseUrl = `${window.location.protocol}//${window.location.host}/`;
ort.env.wasm.wasmPaths = wasmBaseUrl;
ort.env.wasm.numThreads = 4;

const MODELS = {
  sam_b_int8: [
    {
      name: "sam-b-encoder-int8",
      url: "/sam_vit_b-encoder-int8.onnx",
      size: 108,
    },
    {
      name: "sam-b-decoder-int8",
      // url: "/sam_vit_b-decoder-int8.onnx",
      url: "/sam_vit_b-decoder-int8.onnx",
      size: 5,
    },
  ],
};
const config = {
  model: "sam_b_int8",
  provider: "wasm",
  device: "gpu",
  threads: 4,
};
let points: number[][] = [];
let labels: number[] = [];
var imageImageData: any;
var maskImageData: any;
var image_embeddings: any;
const MODEL_WIDTH = 1024;
const MODEL_HEIGHT = 1024;
var isClicked = false;
let filein;
// the image size on canvas
const MAX_WIDTH = 500;
const MAX_HEIGHT = 500;

function log(i: any) {
  document.getElementById("status")!.innerText += `\n${i}`;
}

// 添加一个清除缓存的函数
async function clearModelCache() {
  caches.keys().then((keys) => {
    keys.forEach((key) => {
      caches.delete(key);
    });
    console.log("All caches cleared");
  });
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const init = async () => {
    // await clearModelCache();
    const canvas = canvasRef.current!;
    canvas.style.cursor = "wait";

    filein = document.getElementById("file-in");
    const decoder_latency = document.getElementById("decoder_latency");

    document.getElementById("clear-button")!.addEventListener("click", () => {
      points = [];
      labels = [];
      decoder(points, labels);
    });

    let img = document.getElementById("original-image") as HTMLImageElement;

    await load_models(MODELS[config.model as keyof typeof MODELS]).then(
      () => {
        canvas.addEventListener("click", handleClick);
        canvas.addEventListener("mousemove", handleMouseMove);
        document
          .getElementById("cut-button")!
          .addEventListener("click", handleCut);

        // image upload
        filein!.onchange = function (evt) {
          console.log("onchange");

          let target = evt.target || window.event?.src,
            files = target.files;
          if (FileReader && files && files.length) {
            let fileReader = new FileReader();
            fileReader.onload = () => {
              img.onload = () => handleImage(img);
              img.src = fileReader.result;
            };
            fileReader.readAsDataURL(files[0]);
          }
        };
        handleImage(img);
      },
      (e) => {
        log(e);
      }
    );
  };

  /**
   * handler called when image available
   */
  async function handleImage(img: HTMLImageElement) {
    const encoder_latency = document.getElementById("encoder_latency")!;
    encoder_latency.innerText = "";
    points = [];
    labels = [];
    filein.disabled = true;
    decoder_latency.innerText = "";
    const canvas = canvasRef.current!;

    canvas.style.cursor = "wait";
    image_embeddings = undefined;

    let width = img.width;
    let height = img.height;
    if (width > height) {
      if (width > MAX_WIDTH) {
        height = height * (MAX_WIDTH / width);
        width = MAX_WIDTH;
      }
    } else {
      if (height > MAX_HEIGHT) {
        width = width * (MAX_HEIGHT / height);
        height = MAX_HEIGHT;
      }
    }
    width = Math.round(width);
    height = Math.round(height);
    canvas.width = width;
    canvas.height = height;

    var ctx = canvas.getContext("2d")!;
    ctx.drawImage(img, 0, 0, width, height);

    imageImageData = ctx.getImageData(0, 0, width, height);

    const t = await ort.Tensor.fromImage(imageImageData, {
      resizedWidth: MODEL_WIDTH,
      resizedHeight: MODEL_HEIGHT,
    });

    const feed = config.isSlimSam ? { pixel_values: t } : { input_image: t };

    console.log(feed, "feed");

    const session = await MODELS[config.model][0].sess;
    console.log(session, "session");

    const start = performance.now();
    image_embeddings = session.run(feed);
    image_embeddings
      .then((result) => {
        console.log("图像嵌入成功:", result);
        encoder_latency.innerText = `${(performance.now() - start).toFixed(
          1
        )}ms`;
        canvas.style.cursor = "default";
      })
      .catch((error) => {
        console.error("图像嵌入错误:", error);
        log(`编码器错误: ${error.message || JSON.stringify(error)}`);
        canvas.style.cursor = "default";
      });
    filein.disabled = false;
  }

  /*
   * Handle cut-out event
   */
  async function handleCut(event: any) {
    if (points.length == 0) {
      return;
    }
    const canvas = canvasRef.current!;
    const [w, h] = [canvas.width, canvas.height];

    // canvas for cut-out
    const cutCanvas = new OffscreenCanvas(w, h);
    const cutContext = cutCanvas.getContext("2d")!;
    const cutPixelData = cutContext.getImageData(0, 0, w, h);

    // need to rescale mask to image size
    const maskCanvas = new OffscreenCanvas(w, h);
    const maskContext = maskCanvas.getContext("2d")!;
    maskContext.drawImage(await createImageBitmap(maskImageData), 0, 0);
    const maskPixelData = maskContext.getImageData(0, 0, w, h);

    // copy masked pixels to cut-out
    for (let i = 0; i < maskPixelData.data.length; i += 4) {
      if (maskPixelData.data[i] > 0) {
        for (let j = 0; j < 4; ++j) {
          const offset = i + j;
          cutPixelData.data[offset] = imageImageData.data[offset];
        }
      }
    }
    cutContext.putImageData(cutPixelData, 0, 0);

    // Download image
    const link = document.createElement("a");
    link.download = "image.png";
    link.href = URL.createObjectURL(await cutCanvas.convertToBlob());
    link.click();
    link.remove();
  }

  /**
   * handler mouse move event
   */
  async function handleMouseMove(event: any) {
    if (isClicked) {
      return;
    }
    try {
      isClicked = true;
      canvasRef.current!.style.cursor = "wait";
      const point = getPoint(event);
      await decoder([...points, point[0], point[1]], [...labels, 1]);
    } finally {
      canvasRef.current!.style.cursor = "default";
      isClicked = false;
    }
  }

  /**
   * handler to handle click event on canvas
   */
  async function handleClick(event: any) {
    if (isClicked) {
      return;
    }
    const canvas = canvasRef.current!;
    try {
      isClicked = true;
      canvas.style.cursor = "wait";

      const point = getPoint(event);
      const label = 1;
      points.push(point[0]);
      points.push(point[1]);
      labels.push(label);
      await decoder(points, labels);
    } finally {
      canvas.style.cursor = "default";
      isClicked = false;
    }
  }

  function getPoint(event: any) {
    const rect = canvasRef.current!.getBoundingClientRect();
    const x = Math.trunc(event.clientX - rect.left);
    const y = Math.trunc(event.clientY - rect.top);
    return [x, y];
  }

  async function decoder(points: any, labels: any) {
    console.log("decoderdecoderdecoder");

    console.log("decodering", points, labels);
    const canvas = canvasRef.current!;

    let ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = imageImageData.width;
    canvas.height = imageImageData.height;
    ctx.putImageData(imageImageData, 0, 0);

    if (points.length > 0) {
      // need to wait for encoder to be ready
      if (image_embeddings === undefined) {
        await MODELS[config.model as keyof typeof MODELS][0].sess;
      }
      console.log("image_embeddings", image_embeddings);

      // wait for encoder to deliver embeddings
      const emb = await image_embeddings;

      // the decoder
      const session = MODELS[config.model as keyof typeof MODELS][1].sess;

      const feed = feedForSam(emb, points, labels);
      const res = await session.run(feed);

      for (let i = 0; i < points.length; i += 2) {
        ctx.fillStyle = "blue";
        ctx.fillRect(points[i], points[i + 1], 10, 10);
      }
      const mask = res.masks;
      maskImageData = mask.toImageData();

      ctx.globalAlpha = 0.3;
      ctx.drawImage(await createImageBitmap(maskImageData), 0, 0);
    }
  }

  /*
   * create feed for the original facebook model
   */
  function feedForSam(emb: any, points: any, labels: any) {
    const maskInput = new ort.Tensor(
      new Float32Array(256 * 256),
      [1, 1, 256, 256]
    );
    const hasMask = new ort.Tensor(new Float32Array([0]), [1]);
    const origianlImageSize = new ort.Tensor(
      new Float32Array([MODEL_HEIGHT, MODEL_WIDTH]),
      [2]
    );
    const pointCoords = new ort.Tensor(new Float32Array(points), [
      1,
      points.length / 2,
      2,
    ]);
    const pointLabels = new ort.Tensor(new Float32Array(labels), [
      1,
      labels.length,
    ]);

    return {
      image_embeddings: cloneTensor(emb.image_embeddings),
      point_coords: pointCoords,
      point_labels: pointLabels,
      mask_input: maskInput,
      has_mask_input: hasMask,
      orig_im_size: origianlImageSize,
    };
  }

  /**
   * clone tensor
   */
  function cloneTensor(t: any) {
    return new ort.Tensor(t.type, Float32Array.from(t.data), t.dims);
  }

  /*
   * load models one at a time
   */
  async function load_models(models: any) {
    console.log(models, "models");

    const cache = await caches.open("onnx");
    let missing = 0;
    for (const [name, model] of Object.entries(models)) {
      let cachedResponse = await cache.match(model.url);

      if (cachedResponse === undefined) {
        missing += model.size;
      }
    }
    if (missing > 0) {
      log(`downloading ${missing} MB from network ... it might take a while`);
    } else {
      log("loading...");
    }
    const start = performance.now();
    for (const [name, model] of Object.entries(models)) {
      try {
        const opt = {
          executionProviders: [config.provider],
          enableMemPattern: false,
          enableCpuMemArena: false,
          extra: {
            session: {
              disable_prepacking: "1",
              use_device_allocator_for_initializers: "1",
              use_ort_model_bytes_directly: "1",
              use_ort_model_bytes_for_initializers: "1",
            },
          },
        };
        const model_bytes = await fetchAndCache(model.url, model.name);
        const extra_opt = model.opt || {};
        const sess_opt = { ...opt, ...extra_opt };
        console.log(sess_opt, "sess_opt");

        model.sess = await ort.InferenceSession.create(model_bytes, sess_opt);
      } catch (e) {
        log(`${model.url} failed, ${e}`);
      }
    }
    const stop = performance.now();
    log(`ready, ${(stop - start).toFixed(1)}ms`);
  }

  /*
   * fetch and cache url
   */
  async function fetchAndCache(url: any, name: any) {
    try {
      const cache = await caches.open("onnx");
      let cachedResponse = await cache.match(url);
      if (cachedResponse == undefined) {
        await cache.add(url);
        cachedResponse = await cache.match(url);
        log(`${name} (network)`);
      } else {
        log(`${name} (cached)`);
      }
      const data = await cachedResponse?.arrayBuffer();
      return data;
    } catch (error) {
      log(`${name} (network)`);
      return await fetch(url).then((response) => response.arrayBuffer());
    }
  }

  useEffect(() => {
    init();
  }, []);
  return (
    <>
      <title>segment anything example</title>
      <div className="container-fluid">
        <h2>segment anything example</h2>
      </div>
      <br />

      <div style={{ display: "none" }}>
        <img id="original-image" src={"/cp.png"} />
      </div>

      <div>
        <div className="row">
          <div className="col">
            <canvas ref={canvasRef} id="img_canvas"></canvas>
          </div>
          <div className="col">
            <div
              className="rounded-block"
              style={{ marginTop: "40px", maxWidth: "200px" }}
            >
              <h4>Latencies</h4>
              <div style={{ marginTop: "10px" }}>
                encoder: <div id="encoder_latency" className="higlight"></div>
              </div>
              <div style={{ marginTop: "10px" }}>
                decoder: <div id="decoder_latency" className="higlight"></div>
              </div>
            </div>
            <div style={{ marginTop: "40px" }}>
              <form>
                <div className="form-group ">
                  <input
                    title="Upload Image"
                    type="file"
                    id="file-in"
                    name="file-in"
                    accept=".jpg, .png, .jpeg, .gif, .bmp, .tif, .tiff|image/*"
                  />
                </div>
              </form>
              <div className="form-group ">
                <button
                  id="cut-button"
                  type="button"
                  className="btn btn-primary"
                >
                  Cut
                </button>
                <button
                  id="clear-button"
                  type="button"
                  className="btn btn-primary"
                >
                  Clear
                </button>
              </div>
              <div style={{ marginTop: "30px" }}>
                <div>Other providers:</div>
                <a href="index.html?provider=wasm&model=sam_b_int8">wasm</a>
                <a href="index.html?provider=webgpu&model=sam_b">webgpu</a>
              </div>
            </div>

            <div id="status" style={{ font: "1em consolas" }}></div>
          </div>
        </div>
      </div>
    </>
  );
}

export default App;
