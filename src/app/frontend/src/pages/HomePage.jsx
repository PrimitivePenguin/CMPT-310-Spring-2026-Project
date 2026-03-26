import Navbar from "../components/layout/navbar";

export default function HomePage() {
  return (
    <main className="min-h-screen bg-slate-50">
      <Navbar />

      <div className="mx-auto max-w-6xl px-6 py-10">
        <h1 className="text-4xl font-semibold text-slate-900">MoodLens</h1>
        <p className="mt-3 max-w-2xl text-slate-600">
          Facial emotion detection from images using a convolutional neural network.
        </p>
      </div>
    </main>
  );
}