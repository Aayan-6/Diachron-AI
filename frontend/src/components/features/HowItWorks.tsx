import { motion } from 'framer-motion';
import {
  UploadCloud,
  BrainCircuit,
  Binary,
  Cpu,
  Terminal,
  MapPin,
  Sparkles,
} from 'lucide-react';

const pipelineStages = [
  {
    step: '01',
    title: 'GeoTIFF Ingestion',
    tag: 'Input',
    description:
      'Upload optical or SAR Cloud-Optimized GeoTIFFs (COGs) as single scenes or bi-temporal comparison pairs.',
    icon: <UploadCloud className="text-accent-cyan" size={24} />,
    tech: 'GDAL / COG Ingest',
  },
  {
    step: '02',
    title: 'Intent Parsing',
    tag: 'Controller',
    description:
      'Agentic Controller evaluates natural-language query to extract spatial extents, temporal constraints, and target entities.',
    icon: <BrainCircuit className="text-accent-cyan" size={24} />,
    tech: 'Geo-LLM Intent Parser',
  },
  {
    step: '03',
    title: 'GIS Pre-processing',
    tag: 'Raster Engine',
    description:
      'Automated co-registration, radiometric normalization, SAR speckle filtering, and reprojection to EPSG:4326.',
    icon: <Binary className="text-accent-cyan" size={24} />,
    tech: 'Rasterio / NumPy',
  },
  {
    step: '04',
    title: 'Specialist AI Routing',
    tag: 'Inference',
    description:
      'Dynamically routes tasks to Semantic Archive Retrieval, Multi-Temporal ChangeFormer V6, or False-Alarm Quality Suppression.',
    icon: <Cpu className="text-accent-cyan" size={24} />,
    tech: 'ChangeFormer V6 / RemoteCLIP',
  },
  {
    step: '05',
    title: 'Aggregator & Trace Logger',
    tag: 'Auditability',
    description:
      'Calculates quantitative metrics (hectares, IoU confidence) and compiles an immutable, auditable execution trace.',
    icon: <Terminal className="text-accent-cyan" size={24} />,
    tech: 'Verifiable Telemetry',
  },
  {
    step: '06',
    title: 'Evidence Delivery',
    tag: 'Visualization',
    description:
      'Overlays bounding boxes, change masks, and heatmaps directly on interactive map with conversational rationale.',
    icon: <MapPin className="text-accent-cyan" size={24} />,
    tech: 'MapLibre GL / Vector Tile',
  },
];

export const HowItWorks = () => {
  return (
    <section id="how-it-works" className="py-24 bg-space-navy relative overflow-hidden border-t border-white/5 scroll-mt-6">
      {/* Background ambient lighting */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[500px] bg-accent-cyan/5 rounded-full blur-3xl pointer-events-none" />

      <div className="container mx-auto px-6 relative z-10">
        {/* Section Heading */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center max-w-3xl mx-auto mb-16"
        >
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-xs font-mono text-accent-cyan mb-4">
            <Sparkles size={12} />
            <span>Agentic Workflow Architecture</span>
          </div>
          <h2 className="text-3xl md:text-5xl font-bold text-white mb-4 tracking-tight">
            How Diachron AI Operates
          </h2>
          <p className="text-sm md:text-base text-slate-400 leading-relaxed">
            From raw multispectral & SAR rasters to mathematically verifiable intelligence — driven by an autonomous multi-agent pipeline.
          </p>
        </motion.div>

        {/* 6-Stage Pipeline Grid with Staggered Scroll Reveals */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 relative">
          {pipelineStages.map((stage, index) => (
            <motion.div
              key={stage.step}
              initial={{ opacity: 0, y: 25 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-50px' }}
              transition={{ delay: index * 0.1, duration: 0.5 }}
              className="bg-space-black/60 border border-white/10 hover:border-accent-cyan/40 rounded-xl p-6 relative group transition-all duration-300 flex flex-col justify-between shadow-xl"
            >
              {/* Corner Step Counter */}
              <div className="flex items-center justify-between mb-4">
                <div className="w-12 h-12 rounded-xl bg-space-navy border border-white/10 flex items-center justify-center text-accent-cyan group-hover:scale-110 group-hover:border-accent-cyan/40 transition-all duration-300">
                  {stage.icon}
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono uppercase px-2 py-0.5 rounded bg-white/5 text-slate-400 border border-white/10">
                    {stage.tag}
                  </span>
                  <span className="text-xl font-mono font-bold text-white/20 group-hover:text-accent-cyan/40 transition-colors">
                    {stage.step}
                  </span>
                </div>
              </div>

              {/* Title & Description */}
              <div className="mb-4">
                <h3 className="text-lg font-bold text-white mb-2 group-hover:text-accent-cyan transition-colors">
                  {stage.title}
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed">
                  {stage.description}
                </p>
              </div>

              {/* Underlying Framework / Tech Spec */}
              <div className="pt-3 border-t border-white/10 flex items-center justify-between text-[10px] font-mono text-slate-400">
                <span className="text-white/40">TECH:</span>
                <span className="text-accent-cyan/90">{stage.tech}</span>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
