import type { ChatMessage } from '@/types/chat';
import { delay, generateId } from '@/lib/utils';

const knowledgeBase: Record<string, string> = {
  'why did the model predict this class': `The model's prediction is based on learned patterns from the training dataset. It analyzes visual features in the MRI scan — such as texture, shape, intensity patterns, and spatial relationships — to assign probability scores to each class.

The highest-scoring class becomes the prediction, but this is a statistical estimate, not a definitive diagnosis. The confidence score reflects how strongly the model's learned features match the input image, not the medical certainty of the finding.

**Important:** Model predictions should always be reviewed by qualified medical professionals.`,

  'explain the grad-cam heatmap': `**Grad-CAM (Gradient-weighted Class Activation Mapping)** is an explainability technique that highlights which regions of the input image were most influential in the model's classification decision.

**How it works:**
1. The model makes a forward pass to produce class scores
2. Gradients of the target class score are computed with respect to the final convolutional layer
3. These gradients are globally averaged to produce importance weights
4. A weighted combination of feature maps creates the heatmap

**What the colors mean:**
- 🔴 **Red/warm regions** — High influence on the prediction
- 🔵 **Blue/cool regions** — Low influence on the prediction

**What it does NOT tell you:**
- It doesn't outline tumor boundaries
- It doesn't confirm that highlighted regions are abnormal
- The highlighted area may include surrounding context used by the model`,

  'what is the difference between grad-cam and segmentation': `**Grad-CAM** and **Segmentation** serve fundamentally different purposes:

**Grad-CAM (Classification Explanation)**
- Shows which image regions influenced the *classification decision*
- Produces a coarse heatmap overlay
- Answers: "Where did the model look to make its prediction?"
- Does not delineate precise boundaries

**Segmentation (Region Prediction)**
- Predicts pixel-level masks for anatomical or pathological regions
- Produces precise boundary delineation
- Answers: "Where does the model think specific structures are located?"
- Uses a separate model architecture (e.g., U-Net)

Think of Grad-CAM as "why the model decided" and segmentation as "what the model sees."`,

  'explain the class scores in simple terms': `The model outputs a score for each possible class. You can think of these scores as the model's "confidence" in each option:

- Each score ranges from 0 to 1 (or 0% to 100%)
- All scores together sum to approximately 1.0
- The highest score is the model's prediction

**Example interpretation:**
- Glioma: 84.7% → The model considers this the most likely class
- Meningioma: 9.8% → Some features match, but less strongly
- Pituitary: 4.1% → Few matching features
- No Tumor: 1.4% → Very few matching features

**Important caveats:**
- These are NOT calibrated medical probabilities
- A high score means the image strongly matches learned patterns, not that a condition is definitively present
- Always consult medical professionals for clinical interpretation`,

  'what are the limitations of this result': `**Model Limitations:**

1. **Training data bias** — The model learned from a specific dataset that may not represent all patient populations, imaging equipment, or scan protocols

2. **2D slice analysis** — The model analyzes individual 2D slices, missing volumetric context that radiologists use

3. **No clinical context** — The model doesn't consider patient history, symptoms, or other diagnostic information

4. **Confidence ≠ accuracy** — A high confidence score doesn't guarantee correctness; the model can be confidently wrong

5. **Limited class coverage** — Only 4 classes are supported. Other pathologies will be misclassified into these categories

6. **Explanation limitations** — Grad-CAM shows correlation, not causation. Highlighted regions may include artifacts

7. **Not clinically validated** — This is a research/educational tool, not approved for clinical diagnosis

**This tool is designed for research and education. Medical decisions should always involve qualified healthcare professionals.**`,

  'summarize this analysis for a research report': `**Analysis Summary**

This analysis was performed using the NeuroClassifier model (v2.1.0) on the uploaded brain MRI scan.

**Classification Results:**
The model assigned the highest probability score to the predicted class, based on visual features extracted by the ResNet-50 architecture. Individual class probabilities and the confusion between classes should be considered when interpreting the result.

**Explainability:**
Grad-CAM and Grad-CAM++ heatmaps were generated for the top predicted class, highlighting regions of the input that most influenced the classification decision. These should be interpreted as indicators of model attention, not as clinical annotations.

**Segmentation:**
A predicted segmentation mask was generated showing estimated tumor boundaries. This mask represents the model's pixel-level predictions and has not been verified against ground truth for this specific scan.

**Disclaimer:** All results are generated by a machine learning model for research purposes. They have not been reviewed by a medical professional and should not be used for clinical decision-making.`,
};

function findBestResponse(userMessage: string): string {
  const lower = userMessage.toLowerCase().trim();

  for (const [key, value] of Object.entries(knowledgeBase)) {
    if (lower.includes(key) || key.includes(lower.slice(0, 30))) {
      return value;
    }
  }

  // Partial matching
  if (lower.includes('predict') || lower.includes('why')) {
    return knowledgeBase['why did the model predict this class'];
  }
  if (lower.includes('grad-cam') || lower.includes('gradcam') || lower.includes('heatmap')) {
    return knowledgeBase['explain the grad-cam heatmap'];
  }
  if (lower.includes('difference') || lower.includes('segmentation vs') || lower.includes('vs seg')) {
    return knowledgeBase['what is the difference between grad-cam and segmentation'];
  }
  if (lower.includes('score') || lower.includes('confidence') || lower.includes('probability')) {
    return knowledgeBase['explain the class scores in simple terms'];
  }
  if (lower.includes('limitation') || lower.includes('wrong') || lower.includes('accurate') || lower.includes('reliable')) {
    return knowledgeBase['what are the limitations of this result'];
  }
  if (lower.includes('summar') || lower.includes('report')) {
    return knowledgeBase['summarize this analysis for a research report'];
  }

  return `Thank you for your question. As a demo assistant, I can help with these topics:

• Why the model predicted a specific class
• How Grad-CAM heatmaps work
• The difference between Grad-CAM and segmentation
• Understanding class scores
• Model limitations
• Generating analysis summaries

Try asking one of the suggested questions for a detailed response.

*Note: This is a demonstration assistant with pre-defined responses. A production version would use an LLM with full analysis context.*`;
}

export async function mockSendChatMessage(message: string): Promise<ChatMessage> {
  await delay(800 + Math.random() * 700);

  return {
    id: generateId(),
    role: 'assistant',
    content: findBestResponse(message),
    timestamp: new Date().toISOString(),
  };
}

export const suggestedQuestions = [
  'Why did the model predict this class?',
  'Explain the Grad-CAM heatmap.',
  'What is the difference between Grad-CAM and segmentation?',
  'Explain the class scores in simple terms.',
  'What are the limitations of this result?',
  'Summarize this analysis for a research report.',
];
