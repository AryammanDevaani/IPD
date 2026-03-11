import { motion } from "framer-motion";
import { ShieldCheck, ShieldAlert, ShieldQuestion } from "lucide-react";

interface VerificationResultProps {
  status: "verified" | "rejected" | "suspicious";
  confidence: number;
}

const statusConfig = {
  verified: {
    icon: ShieldCheck,
    label: "Verified Authentic",
    colorClass: "text-green-500",
    bgClass: "bg-green-500/10 border-green-500/30",
    barClass: "bg-green-500",
  },
  rejected: {
    icon: ShieldAlert,
    label: "Forgery Detected",
    colorClass: "text-red-500",
    bgClass: "bg-red-500/10 border-red-500/30",
    barClass: "bg-red-500",
  },
  suspicious: {
    icon: ShieldQuestion,
    label: "Requires Review",
    colorClass: "text-yellow-500",
    bgClass: "bg-yellow-500/10 border-yellow-500/30",
    barClass: "bg-yellow-500",
  },
};

const VerificationResult = ({ status, confidence }: VerificationResultProps) => {
  const config = statusConfig[status];
  const Icon = config.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`relative rounded-2xl border p-6 ${config.bgClass} backdrop-blur-xl overflow-hidden shadow-2xl transition-all duration-500`}
    >
      {/* Background glow for the result card */}
      <div className={`absolute top-0 right-0 -mr-16 -mt-16 w-32 h-32 rounded-full blur-[80px] pointer-events-none ${config.bgClass.split(' ')[0]}`} />

      <div className="relative z-10 flex items-center gap-4">
        <div className={`rounded-full p-3 ${config.bgClass}`}>
          <Icon className={`h-8 w-8 ${config.colorClass}`} />
        </div>
        <div className="flex-1">
          <h3 className={`text-xl font-bold ${config.colorClass}`}>{config.label}</h3>
          <p className="text-sm text-slate-400 mt-1">
            AI analysis completed with {confidence}% confidence
          </p>
        </div>
      </div>

      <div className="mt-6 space-y-2">
        <div className="flex justify-between text-sm font-medium text-slate-200">
          <span>Authenticity Score</span>
          <span>{confidence}%</span>
        </div>
        <div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${confidence}%` }}
            transition={{ duration: 1, ease: "easeOut" }}
            className={`h-full ${config.barClass}`}
          />
        </div>
      </div>
    </motion.div>
  );
};

export default VerificationResult;
