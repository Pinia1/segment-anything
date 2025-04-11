import * as ort from "onnxruntime-web/webgpu";

interface ONNXModel {
  name: string;
  url: string;
  sess: any;
  size: number;
  opt: any;
}

export const onnxConfig = {
  model: "sam_b_int8",
  provider: "wasm",
  device: "gpu",
  threads: 4,
};

const MODEL_WIDTH = 1024;
const MODEL_HEIGHT = 1024;
export default class ONNX {
  private model: ONNXModel | undefined;
  private image_embeddings: any;
  private imageImageData: ImageData | undefined;

  constructor(wasmBaseUrl: string, model: ONNXModel) {
    ort.env.wasm.wasmPaths = wasmBaseUrl;
    ort.env.wasm.numThreads = 4;
    this.loadModel(model);
  }
  async loadModel(model: ONNXModel) {
    const opt = {
      executionProviders: [onnxConfig.provider],
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

    const model_bytes = await this.fetchAndCache(model.url);
    const extra_opt = model.opt || {};
    const sess_opt = { ...opt, ...extra_opt };
    model.sess = await ort.InferenceSession.create(model_bytes!, sess_opt);
    this.model = model;
  }

  async fetchAndCache(url: ONNXModel["url"]) {
    try {
      const cache = await caches.open("onnx");
      let cachedResponse = await cache.match(url);
      if (cachedResponse == undefined) {
        await cache.add(url);
        cachedResponse = await cache.match(url);
      }
      const data = await cachedResponse?.arrayBuffer();
      return data;
    } catch (error) {
      return await fetch(url).then((response) => response.arrayBuffer());
    }
  }

  async matchingImage(
    img: HTMLImageElement,
    canvas: HTMLCanvasElement,
    MAX_WIDTH: number,
    MAX_HEIGHT: number
  ) {
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

    this.imageImageData = ctx.getImageData(0, 0, width, height);

    const t = await (ort.Tensor as any).fromImage(this.imageImageData, {
      resizedWidth: MODEL_WIDTH,
      resizedHeight: MODEL_HEIGHT,
    });

    const feed = { input_image: t };

    this.image_embeddings = this.model?.sess.run(feed);
    this.image_embeddings
      .then((result: any) => {
        console.log("图像嵌入成功:", result);
      })
      .catch((error: any) => {
        console.error("图像嵌入错误:", error);
      });
  }

  clearOnnxCache() {
    caches.keys().then((keys) => {
      keys.forEach((key) => {
        if (key.includes("onnx")) {
          caches.delete(key);
        }
      });
      console.log("All caches cleared");
    });
  }
}
