import { ArrowUpTrayIcon } from "@heroicons/react/24/outline";

export default function UploadCard({
  selectedFile,
  previewUrl,
  onFileChange,
  onPredict,
  loading,
}) {

  return (
    <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="mb-5">
        <h2 className="text-xl font-semibold text-slate-900">Upload a photo</h2>
        <p className="mt-1 text-sm text-slate-600">
          Choose a clear image of a face to get started.
        </p>
      </div>

      {!previewUrl ? (
        <label
          htmlFor="image-upload"
          className="flex cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed border-slate-300 bg-slate-50 px-6 py-10 text-center transition hover:border-[#06B6D4] hover:bg-cyan-50/40"
        >
          <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-cyan-50 text-[#06B6D4]">
            <ArrowUpTrayIcon className="h-6 w-6" />
          </div>

          <span className="text-sm font-medium text-slate-900">
            Click to upload an image
          </span>
          <span className="mt-1 text-sm text-slate-500">
            JPG, JPEG, or PNG
          </span>

          <input
            id="image-upload"
            type="file"
            accept="image/png,image/jpeg,image/jpg"
            className="hidden"
            onChange={onFileChange}
          />
        </label>
      ) : (
        <div className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-50">
          <img
            src={previewUrl}
            alt="Selected preview"
            className="max-h-72 w-full object-contain bg-slate-100"
          />
          <div className="border-t border-slate-200 px-4 py-3">
            <label
              htmlFor="image-upload"
              className="cursor-pointer text-sm font-medium text-[#06B6D4] transition hover:text-cyan-600"
            >
              Choose a different image
            </label>
            <input
              id="image-upload"
              type="file"
              accept="image/png,image/jpeg,image/jpg"
              className="hidden"
              onChange={onFileChange}
            />
          </div>
        </div>
      )}

      <div className="mt-4 min-h-6">
        {selectedFile ? (
          <p className="text-sm text-slate-600">
            Selected: <span className="font-medium text-slate-900">{selectedFile.name}</span>
          </p>
        ) : (
          <p className="text-sm text-slate-400">No file selected yet.</p>
        )}
      </div>

      <button
        type="button"
        onClick={onPredict}
        disabled={!selectedFile || loading}
        className="mt-6 inline-flex w-full items-center justify-center rounded-2xl bg-[#06B6D4] px-4 py-3 text-sm font-semibold text-white transition hover:bg-cyan-600 disabled:cursor-not-allowed disabled:bg-slate-300"
      >
        {loading ? "Predicting..." : "Predict emotion"}
      </button>
    </div>
  );
}