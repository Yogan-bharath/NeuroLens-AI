import { Sparkles } from 'lucide-react';

interface SuggestedQuestionsProps {
  questions: string[];
  onSelectQuestion: (question: string) => void;
  disabled?: boolean;
}

export default function SuggestedQuestions({
  questions,
  onSelectQuestion,
  disabled = false,
}: SuggestedQuestionsProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1.5 text-[11px] font-medium text-neuro-text-muted">
        <Sparkles className="w-3 h-3 text-neuro-accent" />
        <span>Suggested Explorations</span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {questions.map((q, idx) => (
          <button
            key={idx}
            disabled={disabled}
            onClick={() => onSelectQuestion(q)}
            className="text-left text-[11px] px-2.5 py-1.5 rounded-lg bg-neuro-surface-200/90 border border-neuro-border text-neuro-text-secondary hover:text-neuro-text-primary hover:border-neuro-accent/40 hover:bg-neuro-surface-300 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
