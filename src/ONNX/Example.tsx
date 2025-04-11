import { useEffect, useRef, useState } from "react";
import ONNX, { ONNXModel } from "./index";
const wasmBaseUrl = `${window.location.protocol}//${window.location.host}/`;

const MODELS = [
  {
    name: "sam-b-encoder-int8",
    url: "/sam_vit_b-encoder-int8.onnx",
    size: 108,
  },
  {
    name: "sam-b-decoder-int8",
    url: "/sam_vit_b-decoder-int8.onnx",
    size: 5,
  },
];
function App() {
  const [loading, setLoading] = useState(true);
  const [modelsLoadingTime, setModelsLoadingTime] = useState(0);
  const [matchingImageTime, setMatchingImageTime] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const onnxRef = useRef<ONNX>(null);

  const init = async () => {
    const start = performance.now();
    onnxRef.current = new ONNX(wasmBaseUrl);
    await onnxRef.current.loadModels(MODELS as ONNXModel[]);
    setModelsLoadingTime(performance.now() - start);
    const imageStart = performance.now();
    await onnxRef.current.matchingImage(
      document.getElementById("original-image") as HTMLImageElement,
      canvasRef.current!,
      1024,
      1024
    );
    setLoading(false);
    setMatchingImageTime(performance.now() - imageStart);
  };

  useEffect(() => {
    init();
  }, []);
  return (
    <>
      {loading ? (
        <div>Loading...</div>
      ) : (
        <h1>
          Loading models time: {(modelsLoadingTime / 1000).toFixed(2)}s
          <br />
          Matching image time: {(matchingImageTime / 1000).toFixed(2)}s
        </h1>
      )}

      <div>
        <img style={{ display: "none" }} id="original-image" src={"/cp.png"} />
      </div>

      <div>
        <div className="row">
          <div className="col">
            <canvas ref={canvasRef} id="img_canvas"></canvas>
          </div>
        </div>
      </div>
    </>
  );
}

export default App;
