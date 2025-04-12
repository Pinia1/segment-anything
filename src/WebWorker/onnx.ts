import * as ort from "onnxruntime-web/webgpu";
import * as Comlink from "comlink";
import { ONNXWorker } from "./worker";
//@ts-ignore
import { debounce } from "lodash";

export interface ONNXModel {
  name: string;
  url: string;
  size: number;
  opt?: any;
  sess?: any;
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
  private imageImageData: ImageData | undefined;
  private points: number[] = [];
  private labels: number[] = [];
  private worker: Worker | undefined;
  private onnxWorker: ONNXWorker;
  private onLoad: () => Promise<void>;
  constructor(
    wasmBaseUrl: string,
    models: ONNXModel[],
    onLoad: () => Promise<void>
  ) {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), {
      type: "module",
    });
    this.onnxWorker = Comlink.wrap(this.worker) as unknown as ONNXWorker;
    this.onLoad = onLoad;
    this.loadModels(models, wasmBaseUrl);
    return this;
  }

  async loadModels(models: ONNXModel[], wasmBaseUrl: string) {
    await this.onnxWorker.loadModels(models, wasmBaseUrl);
    this.onLoad();
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

  get model() {
    return this.onnxWorker.models;
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
    canvas.addEventListener(
      "mousemove",
      debounce(async (e: MouseEvent) => {
        const point = this.getPoint(e, canvas);
        const maskImageData = await this.onnxWorker.decoder(
          [...this.points, point[0], point[1]],
          [...this.labels, 1],
          width,
          height,
          this.imageImageData!
        );
        const ctx = canvas.getContext("2d")!;
        ctx.putImageData(this.imageImageData!, 0, 0);

        ctx.globalAlpha = 0.3;
        ctx.drawImage(await createImageBitmap(maskImageData!), 0, 0);
      }, 50)
    );

    canvas.addEventListener("mouseleave", () => {
      this.points = [];
      this.labels = [];
      ctx.putImageData(this.imageImageData!, 0, 0);
    });

    return await this.onnxWorker.sessionRun(feed);
  }

  getPoint(event: any, canvas: HTMLCanvasElement) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.trunc(event.clientX - rect.left);
    const y = Math.trunc(event.clientY - rect.top);
    return [x, y];
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
