import { useState, useRef, useEffect } from 'react';
import type { ChatMessage as ChatMessageType } from '@/types/chat';
import ChatMessage from './ChatMessage';
import SuggestedQuestions from './SuggestedQuestions';
import { mockSendChatMessage, suggestedQuestions } from '@/services/chatService';
import { generateId } from '@/lib/utils';
import {
  Send,
  Trash2,
  Bot,
  Loader2,
  Sparkles,
} from 'lucide-react';

interface AIChatPanelProps {
  analysisId?: string;
  predictedClass?: string | null;
  className?: string;
}

export default function AIChatPanel({
  predictedClass,
  className = '',
}: AIChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessageType[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: `Hello. I am the NeuroLens Copilot.
${
  predictedClass
    ? `I have context on the current analysis (${predictedClass}). `
    : 'I can assist you with understanding MRI classification, Grad-CAM explanations, and segmentation.'
}

Feel free to ask questions about the model results, explainability methods, or limitations.`,
      timestamp: new Date().toISOString(),
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSend = async (textToSend?: string) => {
    const text = (textToSend || inputValue).trim();
    if (!text || isLoading) return;

    const userMsg: ChatMessageType = {
      id: generateId(),
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await mockSendChatMessage(text);
      setMessages(prev => [...prev, response]);
    } catch {
      setMessages(prev => [
        ...prev,
        {
          id: generateId(),
          role: 'assistant',
          content: 'Unable to retrieve explanation. Please try again.',
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setMessages([
      {
        id: generateId(),
        role: 'assistant',
        content: 'Conversation cleared. How can I help you explore this MRI analysis?',
        timestamp: new Date().toISOString(),
      },
    ]);
  };

  return (
    <div className={`neuro-card flex flex-col h-[520px] lg:h-[600px] overflow-hidden ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-neuro-border">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-neuro-accent to-neuro-cyan flex items-center justify-center text-neuro-bg font-bold">
            <Bot className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <h3 className="text-sm font-semibold text-neuro-text-primary">NeuroLens Copilot</h3>
              <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400">
                Demo
              </span>
            </div>
            <p className="text-[11px] text-neuro-text-muted">Context-aware XAI Explainer</p>
          </div>
        </div>

        <button
          onClick={handleClear}
          className="p-1.5 rounded hover:bg-neuro-surface-200 text-neuro-text-muted hover:text-neuro-text-secondary transition-colors"
          title="Clear Conversation"
          aria-label="Clear Conversation"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto py-3 space-y-3 scrollbar-thin pr-1">
        {messages.map(msg => (
          <ChatMessage key={msg.id} message={msg} />
        ))}

        {isLoading && (
          <div className="flex items-center gap-2 text-xs text-neuro-text-muted py-2 px-3 rounded-lg bg-neuro-surface-200/50 w-fit">
            <Loader2 className="w-3.5 h-3.5 animate-spin text-neuro-accent" />
            <span>Analyzing explanation context...</span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Questions */}
      <div className="pt-2 pb-2 border-t border-neuro-border">
        <SuggestedQuestions
          questions={suggestedQuestions}
          onSelectQuestion={q => handleSend(q)}
          disabled={isLoading}
        />
      </div>

      {/* Input area */}
      <form
        onSubmit={e => {
          e.preventDefault();
          handleSend();
        }}
        className="flex items-center gap-2 pt-2 border-t border-neuro-border"
      >
        <input
          type="text"
          value={inputValue}
          onChange={e => setInputValue(e.target.value)}
          placeholder="Ask a question about the model prediction or heatmap..."
          disabled={isLoading}
          className="neuro-input flex-1 text-xs py-2"
        />
        <button
          type="submit"
          disabled={!inputValue.trim() || isLoading}
          className="neuro-btn-primary p-2 text-xs disabled:opacity-40"
          aria-label="Send message"
        >
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
}
