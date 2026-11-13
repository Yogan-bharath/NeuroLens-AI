import { motion } from 'framer-motion';
import PageHeader from '@/components/layout/PageHeader';
import { ResearchDisclaimer } from '@/components/common/ResearchDisclaimer';
import {
  FileText,
  Upload,
  Brain,
  Sparkles,
  Layers,
  AlertTriangle,
  Lock,
  ChevronRight,
} from 'lucide-react';

export default function DocumentationPage() {
  const sections = [
    {
      icon: Upload,
      title: 'Supported Formats & Ingestion Pipeline',
      content: (
        <div className="space-y-2 text-xs text-neuro-text-secondary leading-relaxed">
          <p>
            The current demonstration pipeline accepts 2D brain MRI cross-sectional slices in standard image formats:
          </p>
          <ul className="list-disc pl-5 space-y-1">
            <li><strong>PNG (.png):</strong> Recommended for lossless slice rendering and pixel fidelity.</li>
            <li><strong>JPEG (.jpg, .jpeg):</strong> Supported up to 10 MB per file.</li>
            <li><strong>Resolution:</strong> Slices are internally normalized and bilinearly scaled to a 256 × 256 pixel tensor input.</li>
          </ul>
          <p className="text-[11px] text-neuro-text-muted pt-1">
            <em>Note:</em> DICOM (.dcm) and NIfTI (.nii, .nii.gz) 3D volumetric series are not claimed as supported in this 2D demonstration interface.
          </p>
        </div>
      ),
    },
    {
      icon: Brain,
      title: 'How to Interpret Class Scores',
      content: (
        <div className="space-y-2 text-xs text-neuro-text-secondary leading-relaxed">
          <p>
            The classification layer evaluates four mutually exclusive classes:
          </p>
          <div className="grid grid-cols-2 gap-2 my-2">
            <div className="p-2 rounded bg-neuro-surface-200">
              <strong className="text-neuro-text-primary">1. Glioma:</strong> Intra-axial glial neoplasms.
            </div>
            <div className="p-2 rounded bg-neuro-surface-200">
              <strong className="text-neuro-text-primary">2. Meningioma:</strong> Extra-axial dural-based patterns.
            </div>
            <div className="p-2 rounded bg-neuro-surface-200">
              <strong className="text-neuro-text-primary">3. Pituitary:</strong> Sellar region neoplasms.
            </div>
            <div className="p-2 rounded bg-neuro-surface-200">
              <strong className="text-neuro-text-primary">4. No Tumor:</strong> Absence of focal lesion patterns.
            </div>
          </div>
          <p>
            Scores reflect the Softmax output across the final logit vector. <strong>Crucially:</strong> Softmax scores represent relative internal confidence within this closed 4-class hypothesis space, not calibrated true medical probability.
          </p>
        </div>
      ),
    },
    {
      icon: Sparkles,
      title: 'Explainable AI: Grad-CAM vs. Grad-CAM++',
      content: (
        <div className="space-y-2 text-xs text-neuro-text-secondary leading-relaxed">
          <p>
            <strong>Grad-CAM (Gradient-weighted Class Activation Mapping):</strong> Calculates the gradient of the score for class <em>c</em> with respect to the final convolutional feature maps. It produces a coarse localization map highlighting discriminative regions.
          </p>
          <p>
            <strong>Grad-CAM++:</strong> Employs a weighted average of positive partial derivatives with respect to feature maps, yielding superior localization for multiple object instances and dispersed abnormal signal patterns.
          </p>
          <div className="p-2.5 rounded bg-neuro-surface-200 border border-neuro-border text-[11px] text-neuro-text-muted">
            <strong className="text-neuro-text-primary">Fundamental Warning:</strong> Heatmaps highlight <em>influence on model output</em>. They do NOT represent histological tumor boundaries or microscopic margins.
          </div>
        </div>
      ),
    },
    {
      icon: Layers,
      title: 'What Segmentation Represents',
      content: (
        <div className="space-y-2 text-xs text-neuro-text-secondary leading-relaxed">
          <p>
            The segmentation model (e.g., U-Net architecture) assigns a predicted class label to individual pixels, separating the Tumor Core from Peritumoral Edema.
          </p>
          <p>
            This is distinct from classification saliency: segmentation generates hypothesized structural boundaries, whereas Grad-CAM explains classifier attention.
          </p>
        </div>
      ),
    },
    {
      icon: AlertTriangle,
      title: 'Known Model Limitations',
      content: (
        <div className="space-y-2 text-xs text-neuro-text-secondary leading-relaxed">
          <ul className="list-disc pl-5 space-y-1">
            <li><strong>Domain Shift:</strong> Scans acquired with different MRI field strengths (e.g. 1.5T vs 3.0T) or non-standard pulse sequences may cause distribution mismatch.</li>
            <li><strong>2D Slice Bias:</strong> Analyzing isolated 2D slices loses anatomical 3D continuity.</li>
            <li><strong>Closed World Assumption:</strong> Pathologies outside the 4 trained classes (e.g., metastatic lesions, abscesses) will be erroneously forced into one of the 4 classes.</li>
          </ul>
        </div>
      ),
    },
    {
      icon: Lock,
      title: 'Privacy & Data Governance',
      content: (
        <div className="space-y-2 text-xs text-neuro-text-secondary leading-relaxed">
          <p>
            In this demonstration implementation, all image handling is performed entirely client-side using temporary browser object URLs. No patient health information (PHI) is retained, transmitted to third-party endpoints, or stored in persistent databases.
          </p>
          <p className="text-[11px] text-neuro-text-muted">
            Do not upload identifiable scans without proper de-identification and institutional review.
          </p>
        </div>
      ),
    },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="max-w-4xl space-y-6"
    >
      <PageHeader
        title="Documentation & Research Methodology"
        subtitle="Detailed architectural background, interpretation guidelines, and safety constraints for NeuroLens AI."
      />

      <div className="space-y-4">
        {sections.map((section, idx) => (
          <div key={idx} className="neuro-card space-y-3">
            <div className="flex items-center gap-2 border-b border-neuro-border pb-3">
              <section.icon className="w-4 h-4 text-neuro-accent flex-shrink-0" />
              <h3 className="text-sm font-semibold text-neuro-text-primary">
                {section.title}
              </h3>
            </div>
            {section.content}
          </div>
        ))}
      </div>

      <ResearchDisclaimer />
    </motion.div>
  );
}
