import * as ort from "onnxruntime-web/webgpu";

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
  private model: ONNXModel[] | undefined;
  private image_embeddings: any;
  private imageImageData: ImageData | undefined;
  private points: number[] = [];
  private labels: number[] = [];

  constructor(wasmBaseUrl: string) {
    ort.env.wasm.wasmPaths = wasmBaseUrl;
    ort.env.wasm.numThreads = 4;
    return this;
  }
  async loadModels(models: ONNXModel[]) {
    for (const [name, model] of Object.entries(models)) {
      try {
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
        console.log(sess_opt, "sess_opt");

        model.sess = await ort.InferenceSession.create(model_bytes!, sess_opt);
      } catch (e) {
        console.log(e, "eeee");
      }
    }
    this.model = models;
    return Promise.resolve(true);
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

  public async matchingImage(
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
    const session = await this.model?.[0].sess;
    this.image_embeddings = session.run(feed);
    canvas.addEventListener("mousemove", async (e) => {
      const point = this.getPoint(e, canvas);
      await this.decoder(
        [...this.points, point[0], point[1]],
        [...this.labels, 1],
        canvas
      );
    });
    return await this.image_embeddings
      .then((result: any) => {
        console.log("图像嵌入成功:", result);
      })
      .catch((error: any) => {
        console.error("图像嵌入错误:", error);
      });
  }

  /**
   * 核心方法
   * 编码解码
   */
  async decoder(points: number[], labels: number[], canvas: HTMLCanvasElement) {
    let ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = this.imageImageData!.width;
    canvas.height = this.imageImageData!.height;
    ctx.putImageData(this.imageImageData!, 0, 0);

    if (points.length > 0) {
      if (this.image_embeddings === undefined) {
        await this.model?.[0].sess;
      }

      // wait for encoder to deliver embeddings
      const emb = await this.image_embeddings;

      // the decoder
      const session = this.model?.[1].sess;

      const feed = this.feedForSam(emb, points, labels);
      const res = await session.run(feed);

      for (let i = 0; i < points.length; i += 2) {
        ctx.fillStyle = "blue";
        ctx.fillRect(points[i], points[i + 1], 10, 10);
      }
      const mask = res.masks;
      const maskImageData = mask.toImageData();

      ctx.globalAlpha = 0.3;
      ctx.drawImage(await createImageBitmap(maskImageData), 0, 0);
    }
  }

  /*
   * create feed for the original facebook model
   */
  feedForSam(emb: any, points: number[], labels: number[]) {
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
      image_embeddings: this.cloneTensor(emb.image_embeddings),
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
  cloneTensor(t: any) {
    return new ort.Tensor(t.type, Float32Array.from(t.data), t.dims);
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
