export interface NormalizedScore {
  label: string;
  score: number;
  percentage: number;
  isTopPrediction?: boolean;
}

/**
 * Robustly extracts and normalizes classification score distributions
 * from any backend analysis response shape (array, nested object, dictionary of probabilities).
 * Prevents runtime `TypeError: ...map is not a function`.
 */
export function extractNormalizedScores(data: any): NormalizedScore[] {
  if (!data) return [];

  // 1. Direct scores array at root level
  if (Array.isArray(data.scores) && data.scores.length > 0) {
    return data.scores.map((s: any) => {
      const score = typeof s.score === 'number' ? s.score : (typeof s.probability === 'number' ? s.probability : 0);
      return {
        label: s.label || s.class_name || s.name || 'Unknown',
        score,
        percentage: typeof s.percentage === 'number' ? s.percentage : +(score * 100).toFixed(1),
        isTopPrediction: Boolean(s.isTopPrediction || s.is_top_prediction),
      };
    });
  }

  // 2. Direct scores_list array
  if (Array.isArray(data.scores_list) && data.scores_list.length > 0) {
    return data.scores_list.map((s: any) => {
      const score = typeof s.score === 'number' ? s.score : 0;
      return {
        label: s.label || s.class_name || 'Unknown',
        score,
        percentage: +(score * 100).toFixed(1),
        isTopPrediction: Boolean(s.isTopPrediction),
      };
    });
  }

  // 3. classification field is an array of score items
  if (Array.isArray(data.classification) && data.classification.length > 0) {
    return data.classification.map((c: any) => {
      const score = typeof c.score === 'number' ? c.score : 0;
      return {
        label: c.label || c.class_name || 'Unknown',
        score,
        percentage: +(score * 100).toFixed(1),
        isTopPrediction: Boolean(c.isTopPrediction),
      };
    });
  }

  // 4. classification field is an object containing scores array or probabilities
  if (data.classification && typeof data.classification === 'object') {
    if (Array.isArray(data.classification.scores) && data.classification.scores.length > 0) {
      return extractNormalizedScores({ scores: data.classification.scores });
    }
    if (Array.isArray(data.classification.scores_list) && data.classification.scores_list.length > 0) {
      return extractNormalizedScores({ scores_list: data.classification.scores_list });
    }
    if (data.classification.probabilities && typeof data.classification.probabilities === 'object') {
      const topPred = data.classification.top_prediction || data.classification.predicted_class;
      return Object.entries(data.classification.probabilities).map(([label, val]) => {
        const score = Number(val) || 0;
        return {
          label,
          score,
          percentage: +(score * 100).toFixed(1),
          isTopPrediction: label.toLowerCase() === (topPred || '').toLowerCase(),
        };
      });
    }
  }

  // 5. classification_details with probabilities dictionary
  if (data.classification_details?.probabilities && typeof data.classification_details.probabilities === 'object') {
    const topPred = data.classification_details.predicted_class;
    return Object.entries(data.classification_details.probabilities).map(([label, val]) => {
      const score = Number(val) || 0;
      return {
        label,
        score,
        percentage: +(score * 100).toFixed(1),
        isTopPrediction: label.toLowerCase() === (topPred || '').toLowerCase(),
      };
    });
  }

  // 6. Direct probabilities dictionary at root
  if (data.probabilities && typeof data.probabilities === 'object') {
    const topPred = data.top_prediction || data.predicted_class || data.predictedClass;
    return Object.entries(data.probabilities).map(([label, val]) => {
      const score = Number(val) || 0;
      return {
        label,
        score,
        percentage: +(score * 100).toFixed(1),
        isTopPrediction: label.toLowerCase() === (topPred || '').toLowerCase(),
      };
    });
  }

  return [];
}

