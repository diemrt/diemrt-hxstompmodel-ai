import './App.css'
import ChatInterface from './components/ChatInterface'

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-dark-200 via-dark-100 to-dark-200">
      <header className="bg-black/30 backdrop-blur-md border-b border-white/10 py-4 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-primary-300 via-primary-400 to-primary-500 bg-clip-text text-transparent">
              HX Stomp Assistant
            </h1>
            <div className="h-8 w-8 rounded-full bg-gradient-to-r from-primary-400 to-primary-600 animate-pulse"></div>
          </div>
        </div>
      </header>
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <ChatInterface />
      </main>
    </div>
  )
}

export default App
