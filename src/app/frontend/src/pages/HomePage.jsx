import Navbar from "../components/layout/navbar";
import HeroSection from "../components/home/HeroSection";

export default function HomePage() {
  return (
    <main className="min-h-screen bg-slate-50">
      <Navbar />
      <HeroSection />
    </main>
  );
}