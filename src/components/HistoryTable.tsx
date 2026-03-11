import { motion } from "framer-motion";
import { ShieldCheck, ShieldAlert, ShieldQuestion, Clock } from "lucide-react";

export interface HistoryEntry {
  id: string;
  name: string;
  status: "verified" | "rejected" | "suspicious";
  confidence: number;
  timestamp: Date;
}

const statusIcons = {
  verified: ShieldCheck,
  rejected: ShieldAlert,
  suspicious: ShieldQuestion,
};

const statusColors = {
  verified: "text-green-500",
  rejected: "text-red-500",
  suspicious: "text-yellow-500",
};

const statusLabels = {
  verified: "Verified",
  rejected: "Rejected",
  suspicious: "Suspicious",
};

const HistoryTable = ({ entries }: { entries: HistoryEntry[] }) => {
  if (entries.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-slate-500">
        <Clock className="mb-3 h-8 w-8 opacity-40" />
        <p className="text-sm">No verifications yet</p>
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-lg border border-slate-800 bg-slate-950/50">
      <table className="w-full text-left text-sm">
        <thead className="bg-slate-900/80 text-slate-400">
          <tr>
            <th className="px-4 py-3 font-medium">Document</th>
            <th className="px-4 py-3 font-medium">Status</th>
            <th className="px-4 py-3 font-medium">Conf</th>
            <th className="px-4 py-3 font-medium text-right">Time</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {entries.map((entry, i) => {
            const Icon = statusIcons[entry.status];
            return (
              <motion.tr
                key={entry.id}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.05 }}
                className="group relative transition-all hover:bg-slate-800/60 text-slate-300 hover:text-white cursor-default"
              >
                <td className="px-4 py-4 font-medium max-w-[150px] truncate relative z-10 border-l-2 border-transparent group-hover:border-cyan-500 transition-colors" title={entry.name}>
                  {entry.name}
                </td>
                <td className="px-4 py-4 relative z-10">
                  <div className="flex items-center gap-2">
                    <Icon className={`h-4 w-4 ${statusColors[entry.status]} group-hover:scale-110 transition-transform`} />
                    <span className="capitalize hidden sm:inline font-medium">{statusLabels[entry.status]}</span>
                  </div>
                </td>
                <td className="px-4 py-4 font-mono font-medium relative z-10">
                  <span className="px-2 py-1 rounded-md bg-white/5 group-hover:bg-white/10 transition-colors">
                    {entry.confidence}%
                  </span>
                </td>
                <td className="px-4 py-4 text-right text-slate-500 group-hover:text-slate-400 transition-colors relative z-10">
                  {entry.timestamp.toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </td>
              </motion.tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default HistoryTable;
