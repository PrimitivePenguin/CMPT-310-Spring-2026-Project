export default function Navbar() {
  return (
    <header className="border-b border-slate-200 bg-white/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <a href="/" className="flex items-center gap-3 cursor-pointer">
          <img
            src="/moodlens-mark.svg"
            alt="MoodLens logo"
            className="h-9 w-9"
          />
          <img
            src="/moodlens-word.svg"
            alt="MoodLens"
            className="h-9 w-auto"
          />
        </a>
      </div>
    </header>
  );
}