import React, { useState } from 'react';
import axios from 'axios';

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);


  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { sender: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);

    try {
      const response = await axios.get('http://localhost:8000/ask', {
        params: { query: input },
      });

      const botMessage = { sender: 'bot', text: response.data.response };
      setMessages(prev => [...prev, botMessage]);
      console.log('Response data:', response.data.plot_path);
      if (response.data.plot_path) {
        const plotMessage = {
          sender: 'bot',
          text: 'Here is the plot:',
          image: `..\..\\${response.data.plot_path}`,
        };
        setMessages(prev => [...prev, plotMessage]);
      }
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = { sender: 'bot', text: 'Sorry, something went wrong.' };
      setMessages(prev => [...prev, errorMessage]);
    }

    setInput('');
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: '600px', margin: 'auto', fontFamily: 'Arial' }}>
      <h2>Viz-Chatbot</h2>
      <div style={{ border: '1px solid #ccc', padding: '10px', height: '400px', overflowY: 'scroll' }}>
        {messages.map((msg, index) => (
          <div key={index} style={{ textAlign: msg.sender === 'user' ? 'right' : 'left', marginBottom: '10px' }}>
            <p><strong>{msg.sender}:</strong> {msg.text}</p>
            {msg.image && <img src={msg.image} alt="Plot" style={{ maxWidth: '100%' }} />}
          </div>
        ))}
        {loading && <p><em>Bot is thinking...</em></p>}
      </div>
      <input
        type="text"
        value={input}
        onChange={e => setInput(e.target.value)}
        onKeyDown={e => e.key === 'Enter' && sendMessage()}
        placeholder="Type your message..."
        style={{ width: '80%', padding: '10px', marginTop: '10px' }}
      />
      <button onClick={sendMessage} style={{ padding: '10px' }}>Send</button>
    </div>
  );
};

export default Chatbot;