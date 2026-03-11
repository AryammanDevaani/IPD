import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import { Shield, Zap } from "lucide-react";
import UploadZone from "@/components/UploadZone";
import VerificationResult from "@/components/VerificationResult";
import HistoryTable, { type HistoryEntry } from "@/components/HistoryTable";
import StatsBar from "@/components/StatsBar";
import { Button } from "@/components/ui/button";

type VerificationStatus = "verified" | "rejected" | "suspicious";

const Dashboard = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [result, setResult] = useState<{
    status: VerificationStatus;
    confidence: number;
  } | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  const simulateVerification = useCallback(() => {
    if (!file) return;
    setIsVerifying(true);
    setResult(null);

    setTimeout(() => {
      const rand = Math.random();
      const status: VerificationStatus =
        rand > 0.4 ? "verified" : rand > 0.15 ? "suspicious" : "rejected";
      const confidence =
        status === "verified"
          ? 85 + Math.floor(Math.random() * 15)
          : status === "suspicious"
          ? 50 + Math.floor(Math.random() * 25)
          : 20 + Math.floor(Math.random() * 30);

      setResult({ status, confidence });
      setHistory((prev) => [
        {
          id: crypto.randomUUID(),
          name: file.name,
          status,
          confidence,
          timestamp: new Date(),
        },
        ...prev,
      ]);
      setIsVerifying(false);
    }, 2500);
  }, [file]);

  return (
    <div className="relative min-h-screen overflow-hidden bg-slate-950 font-sans text-slate-50 selection:bg-cyan-500/30">
      {/* Premium Background Effects */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="absolute top-[-10%] right-[-5%] w-[40%] h-[40%] bg-cyan-500/10 blur-[100px] rounded-full animate-pulse opacity-50 block" />
        <div className="absolute bottom-[-10%] left-[-5%] w-[40%] h-[40%] bg-blue-500/10 blur-[100px] rounded-full animate-pulse opacity-50 block" />
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#8080800a_1px,transparent_1px),linear-gradient(to_bottom,#8080800a_1px,transparent_1px)] bg-[size:24px_24px]" />
      </div>

      <div className="relative z-10 mx-auto max-w-7xl px-4 py-8 md:px-6 lg:py-12">
        <header className="mb-12 flex flex-col items-center justify-between gap-6 md:flex-row lg:mb-16">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center gap-4"
          >
            <div className="relative flex h-16 w-16 items-center justify-center rounded-2xl bg-slate-900 border border-cyan-500/20 shadow-[0_0_20px_rgba(6,182,212,0.15)]">
              <Shield className="h-8 w-8 text-cyan-400" />
            </div>
            <div>
              <h1 className="text-3xl font-bold tracking-tight text-white md:text-3xl">Verification Dashboard</h1>
              <p className="text-slate-400">Intelligent Document Authentication</p>
            </div>
          </motion.div>
          <Button variant="outline" className="gap-2 border-cyan-500/20 bg-cyan-500/5 hover:bg-cyan-500/10">
            <div className="h-2 w-2 animate-pulse rounded-full bg-green-500" />
            System Online
          </Button>
        </header>

        <div className="grid gap-8 lg:grid-cols-12">
          <section className="lg:col-span-12 xl:col-span-8">
            <div className="rounded-[2.5rem] border border-white/5 bg-slate-900/40 p-6 backdrop-blur-2xl md:p-10 shadow-2xl relative overflow-hidden">
              {/* Card internal gradient */}
              <div className="absolute top-0 right-0 p-32 bg-cyan-500/5 blur-[100px] rounded-full pointer-events-none" />
              
              <div className="relative z-10 mb-8 flex items-center justify-between">
                <div>
                  <h2 className="text-3xl font-bold tracking-tight text-white mb-1">Deep Scan Analysis</h2>
                  <p className="text-slate-400">Verify authenticity with deep-scan AI and multi-layered analysis</p>
                </div>
                <div className="p-3 bg-yellow-500/10 rounded-xl border border-yellow-500/20 hidden sm:block">
                  <Zap className="h-6 w-6 text-yellow-500" />
                </div>
              </div>
              <div className="relative z-10">
                <UploadZone onFileSelect={setFile} isVerifying={isVerifying} onClear={() => {setFile(null); setResult(null);}} />
              </div>
              <div className="relative z-10 mt-8 flex flex-col gap-6 items-center">
                <Button onClick={simulateVerification} disabled={!file || isVerifying} className="h-14 max-w-sm w-full rounded-2xl text-lg font-bold bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white border-0 shadow-[0_0_30px_rgba(6,182,212,0.3)] hover:shadow-[0_0_40px_rgba(6,182,212,0.5)] transition-all">
                  {isVerifying ? "Analyzing Architecture..." : "Run AI Verification"}
                </Button>
                <div className="w-full">
                  {result && <VerificationResult status={result.status} confidence={result.confidence} />}
                </div>
              </div>
            </div>
          </section>
          <aside className="lg:col-span-12 xl:col-span-4 flex flex-col gap-8">
            <StatsBar entries={history} />
            <div className="rounded-3xl border border-white/5 bg-slate-900/40 p-6 backdrop-blur-2xl shadow-xl flex-1 flex flex-col h-full max-h-[600px] overflow-hidden">
              <h2 className="mb-6 text-xl font-bold text-white flex items-center gap-2">
                 <Shield className="w-5 h-5 text-cyan-400" />
                 Verification Log
              </h2>
              <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
                <HistoryTable entries={history} />
              </div>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
