import { motion } from 'framer-motion';
import { Card } from '../ui/Card';
import { Eye, Target, Layers, Zap, ArrowUpRight } from 'lucide-react';

const capabilities = [
  {
    title: 'Semantic Archive Retrieval',
    description:
      'Open-vocabulary zero-shot text search and exemplar-based visual similarity across multi-temporal satellite archives.',
    icon: <Target className="text-accent-cyan" size={24} />,
    input: 'Natural Query / Exemplar AOI',
    output: 'Ranked Locations + Similarity Metric',
    features: ['Open-Vocabulary Search', 'Exemplar Image-to-Image', 'Hybrid Metadata Filters'],
  },
  {
    title: 'Multi-Temporal Change Detection',
    description:
      'ChangeFormer V6 transformer architecture coupled with IR-MAD to detect physical alterations, onset dates, and evolution trajectories.',
    icon: <Layers className="text-accent-cyan" size={24} />,
    input: 'Multi-Epoch Scene Stacks',
    output: 'Calibrated Vector Polygons + Onset Date',
    features: ['Earliest Onset Date Pinpointing', 'Infrastructure Taxonomy', 'Trajectory Tracking'],
  },
  {
    title: 'False-Alarm Quality Suppression',
    description:
      'Active mitigation of seasonal phenology, cast shadows, and thin cirrus using SCL cloud masks, radiometric normalization, and confidence calibration.',
    icon: <Zap className="text-accent-cyan" size={24} />,
    input: 'Scene Classification (SCL) + Bands',
    output: 'High-Precision Verified Changes',
    features: ['Phenology Decoupling', 'Illumination Normalization', 'Confidence Calibration'],
  },
  {
    title: 'Analyst Triage & Provenance',
    description:
      'Ranked review queue with side-by-side evidence, human-in-the-loop triage, and full RFC 7946 GeoJSON and W3C PROV export.',
    icon: <Eye className="text-accent-cyan" size={24} />,
    input: 'Analyst Decisions (Accept / Reject)',
    output: 'Verifiable Intelligence Dossier',
    features: ['Side-by-Side Timeline Scrubber', 'Human-in-the-Loop Triage', 'W3C PROV Lineage'],
  },
];

export const Capabilities = () => {
  return (
    <section id="capabilities" className="py-24 bg-space-black relative border-t border-white/5">
      <div className="container mx-auto px-6">
        <div className="text-center mb-16 max-w-2xl mx-auto">
          <h2 className="text-3xl md:text-5xl font-bold text-white mb-4 tracking-tight">
            Specialist Model Fleet
          </h2>
          <p className="text-sm md:text-base text-slate-400 leading-relaxed">
            The Agentic Controller orchestrates an ensemble of task-specific neural networks calibrated for Earth observation intelligence.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {capabilities.map((cap, index) => (
            <motion.div
              key={cap.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1, duration: 0.4 }}
              className="h-full"
            >
              <Card variant="glass" className="h-full group hover:border-accent-cyan/50 flex flex-col justify-between transition-all duration-300">
                <div>
                  <div className="flex items-center justify-between mb-6">
                    <div className="p-3 rounded-xl bg-accent-cyan/10 border border-accent-cyan/20 w-fit group-hover:scale-110 group-hover:bg-accent-cyan/20 transition-all duration-300">
                      {cap.icon}
                    </div>
                    <ArrowUpRight size={18} className="text-white/20 group-hover:text-accent-cyan transition-colors" />
                  </div>

                  <h3 className="text-lg font-bold text-white mb-2 group-hover:text-accent-cyan transition-colors">
                    {cap.title}
                  </h3>
                  <p className="text-xs text-slate-400 mb-6 leading-relaxed">
                    {cap.description}
                  </p>
                </div>

                <div className="space-y-3 pt-4 border-t border-white/10">
                  <div className="space-y-1.5 text-[10px] font-mono">
                    <div className="flex justify-between text-slate-400">
                      <span className="text-white/40">IN:</span>
                      <span className="text-slate-300 truncate max-w-[140px]">{cap.input}</span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span className="text-white/40">OUT:</span>
                      <span className="text-accent-cyan truncate max-w-[140px]">{cap.output}</span>
                    </div>
                  </div>

                  <div className="space-y-1 pt-2 border-t border-white/5">
                    {cap.features.map((feature, fIndex) => (
                      <div key={fIndex} className="flex items-center gap-1.5 text-[11px] text-slate-300">
                        <div className="w-1 h-1 rounded-full bg-accent-cyan/80 shrink-0" />
                        <span className="truncate">{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Capabilities;
