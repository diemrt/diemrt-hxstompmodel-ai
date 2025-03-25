import './App.css'
import ChatInterface from './components/ChatInterface'

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-dark-200 to-dark-100">
      <header className="bg-black/20 backdrop-blur-sm border-b border-white/10 p-6 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-primary-400 to-primary-600 bg-clip-text text-transparent">
            HX Stomp Assistant
          </h1>
        </div>
      </header>
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <ChatInterface />
      </main>
    </div>
  )
}

export default App
