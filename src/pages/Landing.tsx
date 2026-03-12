import { motion } from 'framer-motion';
import { Shield, FileCheck, Search, UploadCloud, Cpu, Award } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';

const Landing = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 font-sans selection:bg-cyan-500/30">
      {/* Background Gradients */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-20%] left-[20%] w-[50%] h-[50%] bg-cyan-500/10 blur-[150px] rounded-full animate-pulse" />
        <div className="absolute bottom-[-20%] right-[10%] w-[50%] h-[50%] bg-blue-600/10 blur-[150px] rounded-full" />
        <div className="absolute top-[40%] left-[-10%] w-[40%] h-[40%] bg-teal-500/10 blur-[150px] rounded-full" />
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 mix-blend-overlay" />
      </div>

      <div className="relative z-10 pt-4">
        {/* Navigation */}
        <nav className="flex items-center justify-between px-6 py-4 max-w-7xl mx-auto sticky top-4 z-50 rounded-2xl bg-slate-900/60 backdrop-blur-xl border border-white/10 shadow-lg w-[calc(100%-2rem)]">
          <div className="flex items-center gap-2">
            <Shield className="w-8 h-8 text-cyan-400" />
            <span className="text-xl font-bold tracking-tight">DocVerify</span>
          </div>
          <Button onClick={() => navigate('/verify')} className="bg-cyan-500 hover:bg-cyan-600 text-slate-950 font-bold px-6 py-2 rounded-full transition-all hover:shadow-[0_0_20px_rgba(6,182,212,0.4)]">
            Get Started
          </Button>
        </nav>

        {/* Hero Section */}
        <main className="max-w-7xl mx-auto px-6 pt-20 pb-32 text-center md:text-left flex flex-col md:flex-row items-center gap-12">
          <div className="flex-1 space-y-8">
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-sm font-medium"
            >
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
              Powered by OpenCV
            </motion.div>
            
            <motion.h1 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="text-5xl md:text-7xl font-extrabold tracking-tight leading-tight"
            >
              Verify Docs <br/>
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-teal-300 to-blue-500 drop-shadow-[0_0_30px_rgba(34,211,238,0.4)]">
                In Seconds
              </span>
            </motion.h1>

            <motion.p 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-lg md:text-xl text-slate-400 max-w-2xl leading-relaxed"
            >
              AI-powered document verification using computer vision. Detect tampering, validate authenticity, and process documents at lightning speed.
            </motion.p>

            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="flex flex-col sm:flex-row items-center gap-4 pt-4 justify-center md:justify-start"
            >
              <Button onClick={() => navigate('/verify')} className="w-full sm:w-auto bg-white hover:bg-slate-200 text-slate-900 text-lg font-bold px-8 py-6 rounded-2xl shadow-[0_0_40px_rgba(255,255,255,0.15)] hover:shadow-[0_0_60px_rgba(255,255,255,0.25)] transition-all">
                Start Verifying
              </Button>
              <Button variant="outline" className="w-full sm:w-auto border-white/10 bg-white/5 hover:bg-white/10 text-white text-lg px-8 py-6 rounded-2xl backdrop-blur-md transition-all">
                View Capabilities
              </Button>
            </motion.div>
          </div>

          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 0.8 }}
            className="flex-1 w-full max-w-lg relative"
          >
            {/* Abstract UI Representation */}
            <div className="aspect-square rounded-[3rem] border border-white/10 bg-slate-900/40 backdrop-blur-2xl relative overflow-hidden flex items-center justify-center p-8 shadow-[0_0_100px_rgba(0,0,0,0.5)]">
              <div className="absolute inset-0 bg-gradient-to-tr from-cyan-500/10 to-transparent" />
              
              {/* Scanning Box Simulation */}
              <div className="relative w-full h-full border border-cyan-500/30 rounded-2xl flex flex-col p-6 gap-4">
                 {/* Corner decorations */}
                 <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-cyan-400" />
                 <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-cyan-400" />
                 <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-cyan-400" />
                 <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-cyan-400" />
                 
                 {/* Background Grid Pattern */}
                 <div className="absolute inset-0 bg-[linear-gradient(to_right,#06b6d41a_1px,transparent_1px),linear-gradient(to_bottom,#06b6d41a_1px,transparent_1px)] bg-[size:20px_20px] rounded-2xl pointer-events-none" />

                 <div className="w-24 h-24 rounded-xl bg-slate-800/80 animate-pulse border border-white/5 shadow-inner" />
                 <div className="h-4 w-3/4 bg-slate-800/80 rounded mt-4 border border-white/5" />
                 <div className="h-4 w-1/2 bg-slate-800/80 rounded border border-white/5" />
                 
                 {/* Scanning laser with glow trailing effect */}
                 <motion.div 
                    animate={{ y: [0, 110, 0] }}
                    transition={{ repeat: Infinity, duration: 4, ease: "easeInOut" }}
                    className="absolute top-0 left-0 right-0 h-[2px] bg-cyan-400 shadow-[0_4px_20px_3px_rgba(34,211,238,0.6)] z-10"
                 >
                    <div className="absolute inset-0 bg-gradient-to-b from-cyan-400/50 to-transparent h-8 -top-8 pointer-events-none" />
                 </motion.div>
              </div>
            </div>
          </motion.div>
        </main>

        {/* Features Section */}
        <section id="features" className="py-32 bg-slate-950 border-t border-white/5 relative z-10">
          <div className="max-w-7xl mx-auto px-6">
            <h2 className="text-3xl md:text-5xl font-bold text-center mb-16">Enterprise-Grade <span className="text-cyan-400">Capabilities</span></h2>
            <div className="grid md:grid-cols-3 gap-8">
              {[
                { icon: Search, title: "Edge Processing", desc: "Advanced contour detection ensuring document boundaries are exact and untampered." },
                { icon: Shield, title: "Tamper Detection", desc: "Pixel-level analysis to catch modifications and digital manipulation instantly." },
                { icon: FileCheck, title: "Watermark Validation", desc: "Deep scanning of physical and digital security marks for flawless validation." }
              ].map((f, i) => (
                <motion.div 
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.1 }}
                  className="bg-slate-900/40 border border-white/10 rounded-[2rem] p-8 hover:bg-slate-800/60 transition-all hover:-translate-y-1 hover:shadow-2xl hover:shadow-cyan-500/5 group"
                >
                  <div className="w-14 h-14 rounded-2xl bg-slate-800 border border-white/5 flex items-center justify-center mb-6 group-hover:scale-110 group-hover:bg-cyan-500/10 group-hover:border-cyan-500/20 transition-all">
                    <f.icon className="w-7 h-7 text-slate-300 group-hover:text-cyan-400 transition-colors" />
                  </div>
                  <h3 className="text-xl font-bold mb-3 text-white">{f.title}</h3>
                  <p className="text-slate-400 leading-relaxed text-sm">{f.desc}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* How It Works Section */}
        <section id="how-it-works" className="py-32 border-t border-white/5 relative bg-slate-900/20">
          <div className="max-w-7xl mx-auto px-6">
            <h2 className="text-3xl md:text-5xl font-bold text-center mb-16">How It <span className="text-cyan-400">Works</span></h2>
            <div className="flex flex-col md:flex-row gap-8 relative">
              {/* Connecting Line */}
              <div className="hidden md:block absolute top-1/2 left-0 right-0 h-0.5 bg-gradient-to-r from-cyan-500/10 via-teal-500/50 to-cyan-500/10 -translate-y-1/2 z-0" />
              
              {[
                { icon: UploadCloud, title: "1. Upload", desc: "Securely upload your ID, passport, or any document via our encrypted portal." },
                { icon: Cpu, title: "2. AI Analysis", desc: "Our engine performs edge detection, barcode reading, and tamper analysis." },
                { icon: Award, title: "3. Verified", desc: "Receive an instant trust score and a detailed report of the document's authenticity." }
              ].map((step, i) => (
                <motion.div 
                  key={i}
                  initial={{ opacity: 0, scale: 0.95 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.15 }}
                  className="flex-1 relative z-10 flex flex-col items-center text-center p-8 bg-slate-950 border border-slate-800 rounded-3xl shadow-xl"
                >
                  <div className="w-20 h-20 rounded-full bg-slate-900 border-2 border-teal-500/30 flex items-center justify-center mb-6 shadow-[0_0_30px_rgba(20,184,166,0.15)]">
                    <step.icon className="w-10 h-10 text-teal-400" />
                  </div>
                  <h3 className="text-2xl font-bold mb-3">{step.title}</h3>
                  <p className="text-slate-400 leading-relaxed">{step.desc}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        <section id="stats" className="py-32 border-t border-white/5 relative overflow-hidden">
  <div className="absolute inset-0 bg-cyan-950/20" />
  <div className="max-w-7xl mx-auto px-6 relative z-10 text-center">
    
    <div className="grid grid-cols-2 gap-12 text-center max-w-2xl mx-auto">
      {[
        { value: "< 2s", label: "Processing Time" },
        { value: "50+", label: "Document Types" }
      ].map((s, i) => (
        <div key={i}>
          <div className="text-5xl font-extrabold text-white mb-2">{s.value}</div>
          <div className="text-cyan-400 font-medium tracking-wide uppercase">{s.label}</div>
        </div>
      ))}
    </div>
 
  </div>
</section>

        {/* Footer */}
        <footer className="py-12 border-t border-slate-900 text-center text-slate-500">
          <p>© 2026 DocVerify AI. All rights reserved.</p>
        </footer>
      </div>
    </div>
  );
};

export default Landing;
