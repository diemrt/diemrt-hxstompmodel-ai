import { useState, useRef, useEffect } from 'react';
import { useMutation, QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import axios from 'axios';

interface ChatMessage {
  role: 'user' | 'assistant' | 'error';
  content: string;
}

const queryClient = new QueryClient({
  defaultOptions: {
    mutations: {
      retry: 2,
      retryDelay: 1000,
    },
  },
});

function ChatComponent() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const chatMutation = useMutation({
    mutationFn: async (question: string) => {
      try {
        const response = await axios.post('http://127.0.0.1:8000/api/chat/', { question });
        return response.data;
      } catch (error) {
        if (axios.isAxiosError(error)) {
          if (error.code === 'ERR_CONNECTION_REFUSED') {
            throw new Error('Cannot connect to the server. Please ensure the backend server is running.');
          }
          throw new Error(error.response?.data?.error || error.message);
        }
        throw error;
      }
    },
    onSuccess: (data) => {
      setMessages(prev => [...prev, { role: 'assistant', content: data.answer }]);
    },
    onError: (error: Error) => {
      setMessages(prev => [...prev, { 
        role: 'error', 
        content: `Error: ${error.message}. Please make sure the backend server is running at http://127.0.0.1:8000`
      }]);
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    setMessages(prev => [...prev, { role: 'user', content: input }]);
    chatMutation.mutate(input);
    setInput('');
  };

  return (
    <div className="flex flex-col h-[calc(100vh-12rem)] bg-black/40 rounded-2xl backdrop-blur-md border border-white/10 shadow-xl overflow-hidden">
      <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-thin scrollbar-thumb-primary-600 scrollbar-track-transparent">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full space-y-4 text-white/50">
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-primary-400 to-primary-600 animate-pulse"></div>
            <p className="text-lg">Ask me anything about the HX Stomp...</p>
          </div>
        )}
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}
          >
            <div
              className={`group relative p-6 rounded-2xl max-w-[80%] shadow-lg transition-all duration-200 ${
                message.role === 'user' 
                  ? 'bg-primary-600 text-white shadow-primary-600/20' 
                  : message.role === 'error'
                    ? 'bg-red-600/80 text-white shadow-red-600/20'
                    : 'bg-white/10 text-white backdrop-blur-sm hover:bg-white/15 shadow-white/5'
              }`}
            >
              <div className={`prose prose-invert prose-sm max-w-none ${
                message.role === 'assistant' 
                  ? 'prose-headings:text-primary-300 prose-strong:text-primary-300 prose-a:text-primary-300 hover:prose-a:text-primary-200 prose-li:my-0 prose-p:my-2 prose-ul:my-2 prose-ol:my-2' 
                  : ''
              }`}>
                {message.role === 'assistant' ? (
                  <ReactMarkdown 
                    remarkPlugins={[remarkGfm]}
                    components={{
                      h3: ({node, ...props}) => <h3 className="text-lg font-semibold mb-2 mt-4" {...props} />,
                      ul: ({node, ...props}) => <ul className="my-2 space-y-1" {...props} />,
                      ol: ({node, ...props}) => <ol className="my-2 space-y-1 list-decimal list-inside" {...props} />,
                      li: ({node, ...props}) => <li className="ml-4" {...props} />,
                      p: ({node, ...props}) => <p className="my-2" {...props} />,
                      code: ({node, ...props}) => <code className="bg-black/20 rounded px-1.5 py-0.5" {...props} />
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                ) : (
                  <p>{message.content}</p>
                )}
              </div>
              {message.role === 'assistant' && (
                <div className="absolute -left-2 top-6 w-1 h-8 bg-primary-500 rounded opacity-0 group-hover:opacity-100 transition-opacity"></div>
              )}
            </div>
          </div>
        ))}
        {chatMutation.isPending && (
          <div className="flex justify-start animate-fade-in">
            <div className="bg-white/10 text-white backdrop-blur-sm p-6 rounded-2xl max-w-[80%] shadow-lg">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-primary-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                <div className="w-2 h-2 bg-primary-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                <div className="w-2 h-2 bg-primary-400 rounded-full animate-bounce"></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <form onSubmit={handleSubmit} className="p-4 border-t border-white/10 bg-black/20">
        <div className="flex gap-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="flex-1 p-4 rounded-xl bg-white/5 text-white placeholder-white/40 border border-white/10 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all focus:bg-white/10"
            placeholder="Ask about HX Stomp..."
            disabled={chatMutation.isPending}
          />
          <button
            type="submit"
            disabled={chatMutation.isPending}
            className="px-6 py-4 bg-primary-600 rounded-xl text-white font-medium hover:bg-primary-500 disabled:opacity-50 disabled:hover:bg-primary-600 transition-all focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 focus:ring-offset-dark-200 shadow-lg shadow-primary-600/20"
          >
            {chatMutation.isPending ? 'Sending...' : 'Send'}
          </button>
        </div>
      </form>
    </div>
  );
}

export default function ChatInterface() {
  return (
    <QueryClientProvider client={queryClient}>
      <ChatComponent />
    </QueryClientProvider>
  );
}