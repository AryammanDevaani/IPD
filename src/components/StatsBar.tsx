import { motion } from "framer-motion";
import { ShieldCheck, ShieldAlert, FileSearch, Activity } from "lucide-react";
import type { HistoryEntry } from "./HistoryTable";

const StatsBar = ({ entries }: { entries: HistoryEntry[] }) => {
  const total = entries.length;
  const verified = entries.filter((e) => e.status === "verified").length;
  const rejected = entries.filter((e) => e.status === "rejected").length;
  const avgConf = total
    ? Math.round(entries.reduce((s, e) => s + e.confidence, 0) / total)
    : 0;

  const stats = [
    { label: "Total Scans", value: total, icon: FileSearch, color: "text-cyan-500" },
    { label: "Verified", value: verified, icon: ShieldCheck, color: "text-green-500" },
    { label: "Rejected", value: rejected, icon: ShieldAlert, color: "text-red-500" },
    { label: "Avg Confidence", value: `${avgConf}%`, icon: Activity, color: "text-cyan-500" },
  ];

  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4 xl:grid-cols-2">
      {stats.map((stat, i) => (
        <motion.div
          key={stat.label}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.1 }}
          className="rounded-xl border border-slate-800 bg-slate-900/50 p-4"
        >
          <div className="flex items-center gap-3">
            <stat.icon className={`h-5 w-5 ${stat.color}`} />
            <div>
              <p className="text-xs text-slate-400">{stat.label}</p>
              <p className="text-xl font-bold font-mono text-slate-100">{stat.value}</p>
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );
};

export default StatsBar;
