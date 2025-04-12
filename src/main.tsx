import { createRoot } from "react-dom/client";
import "./index.css";
import Example from "./WebWorker/Example.tsx";

createRoot(document.getElementById("root")!).render(<Example />);
