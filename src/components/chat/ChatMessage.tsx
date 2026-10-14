import type { ChatMessage as ChatMessageType } from '@/types/chat';
import { Bot, User } from 'lucide-react';
import { formatDate } from '@/lib/utils';

interface ChatMessageProps {
  message: ChatMessageType;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isAssistant = message.role === 'assistant';

  return (
    <div className={`flex gap-3 text-xs ${isAssistant ? 'justify-start' : 'justify-end'}`}>
      {isAssistant && (
        <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-neuro-accent to-neuro-cyan flex items-center justify-center flex-shrink-0 text-neuro-bg">
          <Bot className="w-4 h-4" />
        </div>
      )}

      <div
        className={`max-w-[85%] rounded-xl p-3.5 space-y-1.5 shadow-sm leading-relaxed ${
          isAssistant
            ? 'bg-neuro-surface-200 border border-neuro-border text-neuro-text-primary'
            : 'bg-neuro-accent text-neuro-bg font-medium'
        }`}
      >
        <div className="whitespace-pre-wrap">{message.content}</div>
        <p
          className={`text-[10px] text-right font-mono ${
            isAssistant ? 'text-neuro-text-disabled' : 'text-neuro-bg/70'
          }`}
        >
          {formatDate(message.timestamp)}
        </p>
      </div>

      {!isAssistant && (
        <div className="w-7 h-7 rounded-lg bg-neuro-surface-300 border border-neuro-border flex items-center justify-center flex-shrink-0 text-neuro-text-secondary">
          <User className="w-4 h-4" />
        </div>
      )}
    </div>
  );
}
