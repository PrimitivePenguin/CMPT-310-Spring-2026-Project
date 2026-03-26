import { useEffect, useState } from "react";

import Navbar from "../components/layout/navbar";
import HeroSection from "../components/home/HeroSection";
import UploadCard from "../components/upload/UploadCard";
import Footer from "../components/layout/Footer";

export default function HomePage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [loading, setLoading] = useState(false);

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
  }

  function handlePredict() {
    setLoading(true);

    setTimeout(() => {
      setLoading(false);
    }, 800);
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

            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <h2 className="text-xl font-semibold text-slate-900">Results</h2>
              <p className="mt-2 text-sm text-slate-600">
                Prediction results will appear here once an image is analyzed.
              </p>
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </main>
  );
}