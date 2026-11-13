export type ModalityKey = 'flair' | 't1' | 't1ce' | 't2' | 'seg';

export interface DetectedModality {
  key: ModalityKey;
  name: string;
  fullName: string;
  required: boolean;
  selectedFile: File | null;
  candidates: File[];
  isAmbiguous: boolean;
  isMissing: boolean;
  color: string;
}

export interface FolderInspectionResult {
  folderName: string;
  totalNiftiCount: number;
  allNiftiFiles: File[];
  unmatchedFiles: File[];
  modalities: Record<ModalityKey, DetectedModality>;
  allRequiredPresent: boolean;
  hasAmbiguity: boolean;
}

const MODALITY_CONFIG: Record<ModalityKey, { name: string; fullName: string; required: boolean; color: string }> = {
  flair: {
    name: 'FLAIR',
    fullName: 'Fluid-Attenuated Inversion Recovery',
    required: true,
    color: 'emerald',
  },
  t1: {
    name: 'T1',
    fullName: 'T1-Weighted Structural MRI',
    required: true,
    color: 'blue',
  },
  t1ce: {
    name: 'T1CE',
    fullName: 'T1 Contrast-Enhanced (Gadolinium)',
    required: true,
    color: 'amber',
  },
  t2: {
    name: 'T2',
    fullName: 'T2-Weighted Transverse Relaxation',
    required: true,
    color: 'purple',
  },
  seg: {
    name: 'SEG',
    fullName: 'Ground Truth Annotation (Optional)',
    required: false,
    color: 'slate',
  },
};

/**
 * Determines which modality a NIfTI filename represents using robust regex patterns.
 */
export function classifyNiftiFilename(filename: string): ModalityKey | 'unknown' {
  const name = filename.toLowerCase();
  // Strip .nii or .nii.gz extension
  const base = name.replace(/\.nii(\.gz)?$/, '');

  // 1. Ground Truth Mask / Segmentation
  if (/(seg|mask|label|ground_?truth|truth)/i.test(base)) {
    return 'seg';
  }

  // 2. FLAIR (must be evaluated before T2 because of t2_flair)
  if (/flair/i.test(base)) {
    return 'flair';
  }

  // 3. T1CE / Contrast-Enhanced (must be evaluated before T1)
  if (/(t1_?ce|t1-ce|t1_?gd|t1-gd|t1_?post|t1\+c|t1c|t1_?contrast)/i.test(base)) {
    return 't1ce';
  }

  // 4. T1 (make sure it doesn't match t1ce)
  if (/(^|[\W_])t1(w|pre)?([\W_]|$)/i.test(base) || base.includes('t1')) {
    return 't1';
  }

  // 5. T2 (make sure it doesn't match flair)
  if (/(^|[\W_])t2(w)?([\W_]|$)/i.test(base) || base.includes('t2')) {
    return 't2';
  }

  return 'unknown';
}

/**
 * Recursively inspects a list of files (e.g. from a webkitdirectory input or drag-and-drop)
 * and groups them into detected modalities with ambiguity detection.
 */
export function inspectStudyFolder(files: File[]): FolderInspectionResult {
  const niftiFiles: File[] = [];
  const unmatched: File[] = [];

  // 1. Filter NIfTI files (.nii or .nii.gz)
  for (const file of files) {
    const low = file.name.toLowerCase();
    if (low.endsWith('.nii') || low.endsWith('.nii.gz')) {
      niftiFiles.push(file);
    }
  }

  // Determine study folder name from relative path if available
  let folderName = 'MRI_Study';
  if (niftiFiles.length > 0 && (niftiFiles[0] as any).webkitRelativePath) {
    const parts = (niftiFiles[0] as any).webkitRelativePath.split('/');
    if (parts.length > 1) {
      folderName = parts[0];
    }
  }

  // 2. Classify each NIfTI file into candidate buckets
  const candidates: Record<ModalityKey, File[]> = {
    flair: [],
    t1: [],
    t1ce: [],
    t2: [],
    seg: [],
  };

  for (const file of niftiFiles) {
    const modality = classifyNiftiFilename(file.name);
    if (modality !== 'unknown') {
      candidates[modality].push(file);
    } else {
      unmatched.push(file);
    }
  }

  // 3. Assemble detected modalities report
  const modalities: Record<ModalityKey, DetectedModality> = {} as any;
  let hasAmbiguity = false;
  let allRequiredPresent = true;

  const keys: ModalityKey[] = ['flair', 't1', 't1ce', 't2', 'seg'];
  for (const key of keys) {
    const config = MODALITY_CONFIG[key];
    const candList = candidates[key];
    const isAmbiguous = candList.length > 1;
    const isMissing = config.required && candList.length === 0;

    if (isAmbiguous) hasAmbiguity = true;
    if (isMissing) allRequiredPresent = false;

    modalities[key] = {
      key,
      name: config.name,
      fullName: config.fullName,
      required: config.required,
      selectedFile: candList.length === 1 ? candList[0] : null,
      candidates: candList,
      isAmbiguous,
      isMissing,
      color: config.color,
    };
  }

  return {
    folderName,
    totalNiftiCount: niftiFiles.length,
    allNiftiFiles: niftiFiles,
    unmatchedFiles: unmatched,
    modalities,
    allRequiredPresent,
    hasAmbiguity,
  };
}

/**
 * Recursively scans directory entries from a browser drag-and-drop event.
 */
export async function getFilesFromDataTransferItems(items: DataTransferItemList): Promise<File[]> {
  const files: File[] = [];

  const traverseFileTree = async (item: any): Promise<void> => {
    if (!item) return;

    if (item.isFile) {
      await new Promise<void>(resolve => {
        item.file((f: File) => {
          files.push(f);
          resolve();
        }, () => resolve());
      });
    } else if (item.isDirectory) {
      const dirReader = item.createReader();
      const readEntries = async (): Promise<any[]> => {
        return new Promise(resolve => {
          dirReader.readEntries((entries: any[]) => resolve(entries), () => resolve([]));
        });
      };

      let entries = await readEntries();
      while (entries.length > 0) {
        for (const entry of entries) {
          await traverseFileTree(entry);
        }
        entries = await readEntries();
      }
    }
  };

  const promises: Promise<void>[] = [];
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    const getEntry = (item as unknown as { webkitGetAsEntry?: () => FileSystemEntry | null }).webkitGetAsEntry;
    const entry = typeof getEntry === 'function' ? item.webkitGetAsEntry() : null;
    if (entry) {
      promises.push(traverseFileTree(entry));
    } else {
      const f = item.getAsFile();
      if (f) files.push(f);
    }
  }

  await Promise.all(promises);
  return files;
}

