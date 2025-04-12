import { useEffect, useRef, useState } from "react";
import ONNX, { ONNXModel } from "./onnx";
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
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const onnxRef = useRef<ONNX>(null);

  const init = async () => {
    onnxRef.current = new ONNX(wasmBaseUrl, MODELS, async () => {
      await onnxRef.current!.matchingImage(
        document.getElementById("original-image") as HTMLImageElement,
        canvasRef.current!,
        512,
        512
      );
      setLoading(false);
    });
  };

  useEffect(() => {
    init();
  }, []);

  return (
    <>
      {loading ? <div>Loading...</div> : <h1>1</h1>}

      <div>
        <img style={{ display: "none" }} id="original-image" src={"/dog.jpg"} />
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
