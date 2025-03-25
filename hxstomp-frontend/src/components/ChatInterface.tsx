import { useState } from 'react';
import { useQuery, useMutation, QueryClient, QueryClientProvider } from '@tanstack/react-query';
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
    <div className="flex flex-col h-screen max-w-4xl mx-auto p-4">
      <div className="flex-1 overflow-y-auto mb-4 space-y-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`p-4 rounded-lg ${
              message.role === 'user' 
                ? 'bg-blue-600 ml-auto max-w-[80%] text-white' 
                : message.role === 'error'
                ? 'bg-red-600 max-w-[80%] text-white'
                : 'bg-gray-700 mr-auto max-w-[80%] text-white'
            }`}
          >
            {message.content}
          </div>
        ))}
        {chatMutation.isPending && (
          <div className="bg-gray-700 mr-auto max-w-[80%] p-4 rounded-lg animate-pulse">
            Thinking...
          </div>
        )}
      </div>
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="flex-1 p-2 rounded-lg bg-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="Ask about HX Stomp..."
          disabled={chatMutation.isPending}
        />
        <button
          type="submit"
          disabled={chatMutation.isPending}
          className="px-4 py-2 bg-blue-600 rounded-lg text-white hover:bg-blue-700 disabled:opacity-50"
        >
          {chatMutation.isPending ? 'Sending...' : 'Send'}
        </button>
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