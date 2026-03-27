import { useEffect, useState } from "react";

import Navbar from "../components/layout/navbar";
import HeroSection from "../components/home/HeroSection";
import UploadCard from "../components/upload/UploadCard";
import PredictionCard from "../components/results/PredictionCard";
import Footer from "../components/layout/Footer";
import { predictEmotion } from "../services/api";

export default function HomePage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  function handleFileChange(event) {
    const file = event.target.files?.[0] || null;

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(file);
    setPreviewUrl(file ? URL.createObjectURL(file) : "");
    setPrediction(null);
    setError("");
  }

  async function handlePredict() {
    if (!selectedFile) return;

    setLoading(true);
    setError("");
    setPrediction(null);

    try {
      const startTime = Date.now();

      const result = await predictEmotion(selectedFile);

      const elapsed = Date.now() - startTime;
      const minimumLoadingTime = 500;

      if (elapsed < minimumLoadingTime) {
        await new Promise((resolve) =>
          setTimeout(resolve, minimumLoadingTime - elapsed)
        );
      }

      setPrediction(result);
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-50 flex flex-col">
      <Navbar />
      <HeroSection />

      <section className="pb-16 flex-1">
        <div className="mx-auto max-w-6xl px-6">
          <div className="grid gap-6 lg:grid-cols-2">
            <UploadCard
              selectedFile={selectedFile}
              previewUrl={previewUrl}
              onFileChange={handleFileChange}
              onPredict={handlePredict}
              loading={loading}
            />

            <PredictionCard
              prediction={prediction}
              loading={loading}
              error={error}
            />
          </div>
        </div>
      </section>

      <Footer />
    </main>
  );
}