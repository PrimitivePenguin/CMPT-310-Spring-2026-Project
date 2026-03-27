import { SparklesIcon } from "@heroicons/react/24/outline";
import { emotionMap } from "../../utils/emotionIcons";

export default function PredictionCard({ prediction, loading, error }) {
  const emotionData = prediction ? emotionMap[prediction.emotion] : null;
  const Icon = emotionData?.icon;

  return (
    <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="mb-5">
        <h2 className="text-xl font-semibold text-slate-900">Prediction</h2>
        <p className="mt-1 text-sm text-slate-600">
          MoodLens will show the detected emotion and confidence here.
        </p>
      </div>

      {loading ? (
        <div className="flex min-h-[320px] flex-col items-center justify-center rounded-2xl border border-slate-200 bg-slate-50 text-center">
          <div className="relative mb-5 flex h-14 w-14 items-center justify-center">
            <div className="absolute inset-0 rounded-full border-2 border-cyan-100" />
            <div className="absolute inset-0 animate-spin rounded-full border-2 border-transparent border-t-[#06B6D4]" />
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-cyan-50 text-[#06B6D4]">
              <SparklesIcon className="h-5 w-5" />
            </div>
          </div>

          <p className="text-base font-medium text-slate-900">Analyzing image</p>

          <div className="mt-3 flex items-center gap-1">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-cyan-400 [animation-delay:-0.3s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-cyan-400 [animation-delay:-0.15s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-cyan-400" />
          </div>

          <p className="mt-4 text-sm text-slate-500">
            Just a quick look at the image.
          </p>
        </div>
      ) : error ? (
        <div className="flex min-h-[320px] items-center justify-center rounded-2xl border border-red-200 bg-red-50 p-6 text-center">
          <div>
            <p className="text-base font-semibold text-red-700">Something went wrong</p>
            <p className="mt-2 text-sm text-red-600">{error}</p>
          </div>
        </div>
      ) : prediction ? (
        <div className="min-h-[320px] rounded-2xl border border-slate-200 bg-slate-50 p-6">
          <div className="mb-6 inline-flex rounded-full bg-cyan-50 px-3 py-1 text-xs font-medium text-[#06B6D4]">
            Result
          </div>

          <div>
            <p className="text-sm font-medium text-slate-500">Detected emotion</p>

            <div className="mt-3 flex items-center gap-3">
              {Icon && (
                <div className={`flex h-10 w-10 items-center justify-center rounded-full bg-white ${emotionData.color}`}>
                  <Icon className="h-5 w-5" />
                </div>
              )}

              <h3 className="text-3xl font-semibold text-slate-900 capitalize">
                {prediction.emotion}
              </h3>
            </div>
          </div>

          <div className="mt-8">
            <p className="text-sm font-medium text-slate-500">Confidence</p>
            <p className="mt-2 font-mono text-2xl text-slate-900">
              {typeof prediction.confidence === "number"
                ? `${(prediction.confidence * 100).toFixed(1)}%`
                : prediction.confidence}
            </p>
          </div>

          {prediction.source && (
            <div className="mt-8">
              <p className="text-sm font-medium text-slate-500">Source</p>
              <p className="mt-2 text-sm text-slate-700 capitalize">
                {prediction.source}
              </p>
            </div>
          )}
        </div>
      ) : (
        <div className="flex min-h-[320px] flex-col items-center justify-center rounded-2xl border border-dashed border-slate-300 bg-slate-50 text-center">
          <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-slate-100 text-slate-400">
            <SparklesIcon className="h-6 w-6" />
          </div>
          <p className="text-base font-medium text-slate-900">No prediction yet</p>
          <p className="mt-1 max-w-sm text-sm text-slate-500">
            Upload an image and click predict to see the detected emotion.
          </p>
        </div>
      )}
    </div>
  );
}