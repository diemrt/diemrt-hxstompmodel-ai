import './App.css'
import ChatInterface from './components/ChatInterface'

function App() {
  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <header className="bg-gray-800 p-4 text-center">
        <h1 className="text-2xl font-bold">HX Stomp Assistant</h1>
      </header>
      <main>
        <ChatInterface />
      </main>
    </div>
  )
}

export default App
