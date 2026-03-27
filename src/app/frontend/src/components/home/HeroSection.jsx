export default function HeroSection() {
  return (
    <section className="w-full py-16 bg-gradient-to-b from-white to-[#06B6D4]-50/40">
      <div className="mx-auto max-w-4xl px-6 text-center">
        
        <div className="mb-6 flex justify-center">
          <img
            src="/moodlens-wordmark.svg"
            alt="MoodLens"
            className="h-12 w-auto"
          />
        </div>

        <h1 className="text-4xl font-semibold text-slate-900 sm:text-5xl">
          See what an image <span className="text-[#06B6D4]">feels</span> like
        </h1>

        <p className="mt-4 text-lg text-slate-600">
          Upload a photo of a face and we'll predict the emotion shown
        </p>

        <div className="mt-6">
          <span className="inline-block rounded-full bg-cyan-50 px-3 py-1 text-xs font-medium text-[#06B6D4]">
            Backed by a CNN
          </span>
        </div>

      </div>
    </section>
  );
}