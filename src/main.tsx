import { createRoot } from "react-dom/client";
import "./index.css";
import Example from "./ONNX/Example.tsx";

createRoot(document.getElementById("root")!).render(<Example />);
