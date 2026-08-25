"use client";

import { useEffect, useState } from "react";

type Prediction = {
  label: string;
  confidence: number;
};

type PredictionResponse = {
  prediction: Prediction;
  top_predictions: Prediction[];
  low_confidence: boolean;
  threshold: number;
};

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  "https://plant-disease-atop.onrender.com";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [backendReady, setBackendReady] = useState<boolean | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((res) => res.json())
      .then((data) => setBackendReady(data.model_ready === true))
      .catch(() => setBackendReady(false));
  }, []);

  function handleFileChange(
    event: React.ChangeEvent<HTMLInputElement>
  ) {
    const selected = event.target.files?.[0];

    if (!selected) return;

    setFile(selected);
    setResult(null);
    setError("");
    setPreview(URL.createObjectURL(selected));
  }

  async function analyzeImage() {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("image", file);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to the prediction API."
      );
    } finally {
      setLoading(false);
    }
  }

  function formatConfidence(value: number) {
    return `${(value * 100).toFixed(2)}%`;
  }

  return (
    <main className="min-h-screen bg-green-50 px-6 py-10 text-gray-900">
      <div className="mx-auto max-w-5xl">
        <header className="mb-10 text-center">
          <div className="mb-3 text-5xl">🌿</div>

          <h1 className="text-4xl font-bold text-green-800">
            PlantGuard AI
          </h1>

          <p className="mt-3 text-gray-600">
            AI-powered plant disease classification
          </p>

          <div className="mt-5 inline-flex items-center gap-2 rounded-full bg-white px-4 py-2 text-sm shadow">
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                backendReady === true
                  ? "bg-green-500"
                  : backendReady === false
                  ? "bg-red-500"
                  : "bg-yellow-400"
              }`}
            />

            {backendReady === true
              ? "AI backend online"
              : backendReady === false
              ? "AI backend unavailable"
              : "Checking AI backend..."}
          </div>
        </header>

        <section className="grid gap-8 md:grid-cols-2">
          {/* Upload */}
          <div className="rounded-2xl bg-white p-6 shadow-lg">
            <h2 className="text-2xl font-semibold">
              Analyze a Leaf
            </h2>

            <p className="mt-2 text-sm text-gray-500">
              Upload a plant leaf image to identify its condition.
            </p>

            <label className="mt-6 flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-green-300 bg-green-50 p-8 text-center hover:bg-green-100">
              <span className="text-4xl">📷</span>

              <span className="mt-3 font-medium text-green-800">
                Choose leaf image
              </span>

              <span className="mt-1 text-xs text-gray-500">
                JPG, JPEG, PNG or WebP
              </span>

              <input
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>

            {preview && (
              <div className="mt-6">
                <p className="mb-2 text-sm font-medium">
                  Image preview
                </p>

                <img
                  src={preview}
                  alt="Selected plant leaf"
                  className="max-h-80 w-full rounded-xl object-contain"
                />
              </div>
            )}

            <button
              onClick={analyzeImage}
              disabled={!file || loading || backendReady !== true}
              className="mt-6 w-full rounded-xl bg-green-700 px-5 py-3 font-semibold text-white transition hover:bg-green-800 disabled:cursor-not-allowed disabled:bg-gray-400"
            >
              {loading ? "Analyzing..." : "🔍 Analyze Plant"}
            </button>

            {error && (
              <div className="mt-5 rounded-xl bg-red-50 p-4 text-sm text-red-700">
                <strong>Error:</strong> {error}
              </div>
            )}
          </div>

          {/* Results */}
          <div className="rounded-2xl bg-white p-6 shadow-lg">
            <h2 className="text-2xl font-semibold">
              Prediction
            </h2>

            {!result ? (
              <div className="flex min-h-80 items-center justify-center text-center text-gray-400">
                <div>
                  <div className="text-5xl">🌱</div>
                  <p className="mt-4">
                    Upload an image and click Analyze Plant.
                  </p>
                </div>
              </div>
            ) : (
              <div className="mt-6">
                <div className="rounded-xl bg-green-50 p-5">
                  <p className="text-sm text-gray-500">
                    Most likely condition
                  </p>

                  <h3 className="mt-2 text-2xl font-bold text-green-800">
                    {result.prediction.label}
                  </h3>

                  <p className="mt-3 text-3xl font-bold">
                    {formatConfidence(
                      result.prediction.confidence
                    )}
                  </p>

                  <p className="mt-1 text-sm text-gray-500">
                    Prediction confidence
                  </p>
                </div>

                {result.low_confidence && (
                  <div className="mt-4 rounded-xl bg-yellow-50 p-4 text-sm text-yellow-800">
                    ⚠️ The model confidence is below the configured
                    threshold of{" "}
                    {formatConfidence(result.threshold)}.
                  </div>
                )}

                <div className="mt-7">
                  <h3 className="font-semibold">
                    Top predictions
                  </h3>

                  <div className="mt-3 space-y-3">
                    {result.top_predictions.map(
                      (prediction, index) => (
                        <div
                          key={`${prediction.label}-${index}`}
                          className="rounded-xl border p-4"
                        >
                          <div className="flex justify-between gap-4">
                            <span className="font-medium">
                              #{index + 1} {prediction.label}
                            </span>

                            <span className="font-semibold">
                              {formatConfidence(
                                prediction.confidence
                              )}
                            </span>
                          </div>

                          <div className="mt-2 h-2 overflow-hidden rounded-full bg-gray-200">
                            <div
                              className="h-full bg-green-600"
                              style={{
                                width: `${Math.min(
                                  prediction.confidence * 100,
                                  100
                                )}%`,
                              }}
                            />
                          </div>
                        </div>
                      )
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </section>

        <footer className="mt-10 text-center text-sm text-gray-500">
          PlantGuard AI • EfficientNetB1 • FastAPI • Next.js
        </footer>
      </div>
    </main>
  );
}