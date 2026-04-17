import axios from "axios";
import { useState } from "react";

export default function App() {
  const [result, setResult] = useState(null);

  const predict = async () => {
    const res = await axios.post("YOUR_BACKEND_URL/predict", {
      distance: 120,
      carrier_rating: 4,
      weather_score: 2
    });
    setResult(res.data.delay);
  };

  return (
    <div>
      <h1>AI Logistics Dashboard</h1>
      <button onClick={predict}>Predict</button>
      {result && <p>Delay: {result}</p>}
    </div>
  );
}
