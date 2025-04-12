import * as ort from "onnxruntime-web/webgpu";
import * as Comlink from "comlink";

interface Message {
  type: string;
  data: any;
}

interface ONNXModel {
  name: string;
  url: string;
  size: number;
  opt?: any;
  sess?: any;
}
const MODEL_WIDTH = 1024;
const MODEL_HEIGHT = 1024;
0;
const onnxConfig = {
  model: "sam_b_int8",
  provider: "wasm",
  device: "gpu",
  threads: 4,
};

async function fetchAndCache(url: ONNXModel["url"]) {
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

function toImageData(e2?: any, dims?: any) {
  const t2 = new OffscreenCanvas(dims[2], dims[1]).getContext("2d");
  let n2;
  if (null == t2) throw new Error("Can not access image data");
  {
    let r2, o2, i2;
    void 0 !== (e2 == null ? void 0 : e2.tensorLayout) &&
    "NHWC" === e2.tensorLayout
      ? ((r2 = e2.dims[2]), (o2 = e2.dims[1]), (i2 = e2.dims[3]))
      : ((r2 = e2.dims[3]), (o2 = e2.dims[2]), (i2 = e2.dims[1]));
    const a2 = void 0 !== e2 && void 0 !== e2.format ? e2.format : "RGB",
      s2 = e2 == null ? void 0 : e2.norm;
    let u2, l2;
    void 0 === s2 || void 0 === s2.mean
      ? (u2 = [255, 255, 255, 255])
      : "number" == typeof s2.mean
      ? (u2 = [s2.mean, s2.mean, s2.mean, s2.mean])
      : ((u2 = [s2.mean[0], s2.mean[1], s2.mean[2], 255]),
        void 0 !== s2.mean[3] && (u2[3] = s2.mean[3])),
      void 0 === s2 || void 0 === s2.bias
        ? (l2 = [0, 0, 0, 0])
        : "number" == typeof s2.bias
        ? (l2 = [s2.bias, s2.bias, s2.bias, s2.bias])
        : ((l2 = [s2.bias[0], s2.bias[1], s2.bias[2], 0]),
          void 0 !== s2.bias[3] && (l2[3] = s2.bias[3]));
    const c2 = o2 * r2;
    if (void 0 !== e2) {
      if (void 0 !== e2.height && e2.height !== o2)
        throw new Error(
          "Image output config height doesn't match tensor height"
        );
      if (void 0 !== e2.width && e2.width !== r2)
        throw new Error("Image output config width doesn't match tensor width");
      if (
        (void 0 !== e2.format && 4 === i2 && "RGBA" !== e2.format) ||
        (3 === i2 && "RGB" !== e2.format && "BGR" !== e2.format)
      )
        throw new Error("Tensor format doesn't match input tensor dims");
    }
    const p2 = 4;
    let d2 = 0,
      f2 = 1,
      h = 2,
      g = 3,
      m = 0,
      b = c2,
      y = 2 * c2,
      _ = -1;
    "RGBA" === a2
      ? ((m = 0), (b = c2), (y = 2 * c2), (_ = 3 * c2))
      : "RGB" === a2
      ? ((m = 0), (b = c2), (y = 2 * c2))
      : "RBG" === a2 && ((m = 0), (y = c2), (b = 2 * c2)),
      (n2 = t2.createImageData(r2, o2));
    for (let e3 = 0; e3 < o2 * r2; d2 += p2, f2 += p2, h += p2, g += p2, e3++)
      (n2.data[d2] = (e2.data[m++] - l2[0]) * u2[0]),
        (n2.data[f2] = (e2.data[b++] - l2[1]) * u2[1]),
        (n2.data[h] = (e2.data[y++] - l2[2]) * u2[2]),
        (n2.data[g] = -1 === _ ? 255 : (e2.data[_++] - l2[3]) * u2[3]);
  }
  return n2;
}

/**
 * 处理 ImageData，转换黑色和红色区域为指定颜色
 * @param imageData 原始 ImageData 对象
 * @param blackToColor [r,g,b,a] 黑色区域转换的目标颜色
 * @param redToColor [r,g,b,a] 红色区域转换的目标颜色
 * @param blackThreshold 黑色检测阈值 (0-255)
 * @param redThreshold 红色检测阈值 (0-255)
 * @returns 处理后的 ImageData 对象
 */
function convertColors(
  imageData: ImageData,
  {
    blackToColor = [0, 0, 0, 0], // 默认：黑色变透明 [r,g,b,a]
    redToColor = [0, 255, 0, 255], // 默认：红色变绿色 [r,g,b,a]
    blackThreshold = 30, // 黑色检测阈值
    redThreshold = 100, // 红色检测阈值
  } = {}
) {
  // 遍历所有像素
  for (let i = 0; i < imageData.data.length; i += 4) {
    const r = imageData.data[i];
    const g = imageData.data[i + 1];
    const b = imageData.data[i + 2];

    // 计算亮度
    const brightness = r * 0.299 + g * 0.587 + b * 0.114;

    // 判断是黑色还是红色区域
    if (brightness < blackThreshold) {
      // 黑色区域 -> 转换为指定颜色
      imageData.data[i] = blackToColor[0]; // R
      imageData.data[i + 1] = blackToColor[1]; // G
      imageData.data[i + 2] = blackToColor[2]; // B
      imageData.data[i + 3] = blackToColor[3]; // A
    } else if (r > redThreshold && g < redThreshold && b < redThreshold) {
      // 红色区域 -> 转换为指定颜色
      imageData.data[i] = redToColor[0]; // R
      imageData.data[i + 1] = redToColor[1]; // G
      imageData.data[i + 2] = redToColor[2]; // B
      imageData.data[i + 3] = redToColor[3]; // A
    }
    // 其他颜色保持不变
  }

  return imageData;
}

export class ONNXWorker {
  public models: ONNXModel[] = [];
  private image_embeddings: any;
  constructor() {}
  async loadModels(models: ONNXModel[], wasmBaseUrl: string) {
    ort.env.wasm.wasmPaths = wasmBaseUrl;
    ort.env.wasm.numThreads = 4;
    ort.env.wasm.simd = true;
    for (const [name, model] of Object.entries(models as ONNXModel[])) {
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
          graphOptimizationLevel: "all",
        };
        const model_bytes = await fetchAndCache(model.url);
        const extra_opt = model.opt || {};
        const sess_opt = { ...opt, ...extra_opt };
        model.sess = await ort.InferenceSession.create(model_bytes!, sess_opt);
      } catch (e) {
        console.log(e, "models loading error");
      }
    }
    this.models = models;
    return this.models;
  }

  sessionRun(feed: any) {
    const session = this.models?.[0].sess;
    this.image_embeddings = session.run(feed);
    return this.image_embeddings
      .then((result: any) => {
        console.log("图像嵌入成功:", result);
      })
      .catch((error: any) => {
        console.error("图像嵌入错误:", error);
      });
  }

  async decoder(
    points: number[],
    labels: number[],
    width: number,
    height: number,
    imageImageData: ImageData
  ) {
    const canvas = new OffscreenCanvas(width, height);
    let ctx = canvas.getContext("2d")!;
    canvas.width = imageImageData!.width;
    canvas.height = imageImageData!.height;
    ctx.putImageData(imageImageData!, 0, 0);
    if (points.length > 0) {
      if (this.image_embeddings === undefined) {
        await this.models?.[0].sess;
      }
      // wait for encoder to deliver embeddings
      const emb = await this.image_embeddings;
      // the decoder
      const session = this.models?.[1].sess;
      const feed = this.feedForSam(emb, points, labels);
      const res = await session.run(feed);
      const mask = res.masks;
      const imageData = toImageData(mask, mask.dims);
      return convertColors(imageData, {
        blackToColor: [255, 255, 0, 0], // 黑色变为透明黄色 [r,g,b,a]
        redToColor: [0, 255, 0, 255], // 红色变为不透明绿色 [r,g,b,a]
        blackThreshold: 40, // 可以调整，检测更多黑色区域
        redThreshold: 120, // 可以调整，更精确检测红色
      });
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
}

Comlink.expose(new ONNXWorker());
